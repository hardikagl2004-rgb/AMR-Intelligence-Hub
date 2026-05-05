"""
AI-MRI Hub — Virtual Laboratory Simulation Module
Paste this file alongside your app.py and call render_lab_tab() inside tab10.
"""

import streamlit as st
import streamlit.components.v1 as components
import json, os, math, random
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE BACTERIA DATABASE  (all species the app recognises)
# ─────────────────────────────────────────────────────────────────────────────
BACTERIA_DB = {
    # ── GRAM-NEGATIVE RODS ──────────────────────────────────────────────────
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
    # ── GRAM-POSITIVE COCCI ─────────────────────────────────────────────────
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
    # ── GRAM-POSITIVE RODS ──────────────────────────────────────────────────
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

ANTIBIOTICS = [
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

DILUTIONS = [0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_mri(u_drugs, u_mechs, t_drugs, t_mechs):
    return (u_drugs + u_mechs) / (t_drugs + t_mechs + 1)

def compute_ari(u_mechs, total_genes):
    return u_mechs / (total_genes + 1)

def get_risk(mri):
    if mri < 0.15: return "LOW", "#3B6D11", "🟢"
    if mri < 0.35: return "MODERATE", "#BA7517", "🟡"
    return "HIGH", "#A32D2D", "🔴"

def mic_for_antibiotic(ab, resistance_factor):
    """Compute realistic MIC from resistance factor [0-1]."""
    base = ab["S_break"]
    if resistance_factor > 0.65:
        mic_raw = base * (2 ** (3 + resistance_factor * 5))
    elif resistance_factor > 0.35:
        mic_raw = base * (2 ** (1 + resistance_factor * 3))
    else:
        mic_raw = base * (2 ** (-1 + resistance_factor * 2))
    closest = min(DILUTIONS, key=lambda d: abs(d - mic_raw))
    if closest <= ab["S_break"]:   interp = "S"
    elif closest <= ab["R_break"]: interp = "I"
    else:                          interp = "R"
    return closest, interp

def scan_json_files():
    """Scan cwd for JSON files and map them to known bacteria names."""
    mapping = {}
    for f in os.listdir('.'):
        if not f.endswith('.json'): continue
        fl = f.lower()
        for bname in BACTERIA_DB:
            key = bname.split()[0].lower()  # genus match
            if key in fl:
                mapping[f] = bname
                break
    return mapping

# ─────────────────────────────────────────────────────────────────────────────
# LAB SIMULATION CORE
# ─────────────────────────────────────────────────────────────────────────────
def simulate_lab(bacteria_name, json_file, resistance_override, temperature, ph, incubation_h, cfu_exp, medium):
    db   = BACTERIA_DB[bacteria_name]
    seed = hash(bacteria_name + json_file) % 99999
    rng  = random.Random(seed + int(resistance_override * 1000))

    # ── real gene data from JSON ──
    real_genes, real_u_drugs, real_u_mechs = 0, 0, 0
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
                    if "drug" in cname:      drug_set.add(val)
                    elif "mechanism" in cname: mech_set.add(val)
            except: pass
        real_u_drugs = len(drug_set)
        real_u_mechs = len(mech_set)
    except:
        real_genes   = max(5, int(db["typical_mri"] * 80))
        real_u_drugs = max(2, int(db["typical_mri"] * 10))
        real_u_mechs = max(1, int(db["typical_mri"] * 7))

    # ── lab MRI/ARI ──
    t_drugs = max(real_u_drugs * 3, real_genes // 2)
    t_mechs = max(real_u_mechs * 3, real_genes // 3)
    lab_mri = compute_mri(real_u_drugs, real_u_mechs, t_drugs, t_mechs)
    lab_mri = min(0.99, lab_mri * (0.85 + resistance_override * 0.30))
    lab_ari = compute_ari(real_u_mechs, real_genes)
    lab_ari = min(0.99, lab_ari * (0.9 + resistance_override * 0.20))

    # ── AI prediction (slight noise) ──
    ai_mri  = round(min(0.99, lab_mri  * (0.93 + rng.uniform(0, 0.14))), 3)
    ai_ari  = round(min(0.99, lab_ari  * (0.90 + rng.uniform(0, 0.20))), 3)

    # ── temperature & pH effects on growth ──
    temp_pen = max(0, abs(temperature - db["optimal_temp"]) * 0.06)
    ph_pen   = max(0, abs(ph - db["optimal_ph"]) * 0.15)
    growth_rate = max(0.1, 1.0 - temp_pen - ph_pen + (resistance_override * 0.05))

    # ── growth curve ──
    time_pts = list(range(0, incubation_h + 1, max(1, incubation_h // 20)))
    LAG = max(1, int(2 / growth_rate))
    LOG_DUR = max(4, int(8 * growth_rate))
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

    # ── MIC results ──
    mic_results = []
    for ab in ANTIBIOTICS:
        base_res = db["base_resistance"].get(ab["class"], 0.2)
        res_factor = min(0.98, base_res * (1.0 + resistance_override * 0.8) + rng.uniform(-0.05, 0.05))
        mic_val, mic_interp = mic_for_antibiotic(ab, res_factor)
        # AI reading: slightly different
        ai_res = min(0.98, res_factor * (0.88 + rng.uniform(0, 0.24)))
        ai_mic, ai_interp = mic_for_antibiotic(ab, ai_res)
        mic_results.append({
            "name": ab["name"], "class": ab["class"],
            "lab_mic": mic_val, "lab_interp": mic_interp,
            "ai_mic":  ai_mic,  "ai_interp":  ai_interp,
            "S_break": ab["S_break"], "R_break": ab["R_break"],
        })

    # ── colony count ──
    colony_count = max(5, int(30 * growth_rate + rng.uniform(-5, 10)))

    return {
        "bacteria":     bacteria_name,
        "json_file":    json_file,
        "db":           db,
        "real_genes":   real_genes,
        "real_u_drugs": real_u_drugs,
        "real_u_mechs": real_u_mechs,
        "lab_mri":      round(lab_mri, 3),
        "lab_ari":      round(lab_ari, 3),
        "ai_mri":       ai_mri,
        "ai_ari":       ai_ari,
        "lab_risk":     get_risk(lab_mri),
        "ai_risk":      get_risk(ai_mri),
        "growth_rate":  round(growth_rate, 3),
        "growth_times": [str(t) + "h" for t in time_pts],
        "growth_cfu":   growth_cfu,
        "colony_count": colony_count,
        "mic_results":  mic_results,
        "temperature":  temperature,
        "ph":           ph,
        "incubation_h": incubation_h,
        "cfu_exp":      cfu_exp,
        "medium":       medium,
        "resistance_override": resistance_override,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HTML RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def build_lab_html(sim):
    db       = sim["db"]
    gram_col = db["gram_color"]
    colony_c = db["colony_color"]
    risk_lab = sim["lab_risk"]
    risk_ai  = sim["ai_risk"]

    # ── grow chart data ──
    growth_labels_js = json.dumps(sim["growth_times"])
    growth_data_js   = json.dumps([round(v / 1e6, 2) for v in sim["growth_cfu"]])

    # ── MIC chart data ──
    ab_names  = [r["name"] for r in sim["mic_results"]]
    lab_mics  = [math.log2(max(r["lab_mic"], 0.03)) for r in sim["mic_results"]]
    ai_mics   = [math.log2(max(r["ai_mic"],  0.03)) for r in sim["mic_results"]]

    # ── profile chart data (resistance factor per class) ──
    classes   = list(dict.fromkeys([r["class"] for r in sim["mic_results"]]))
    lab_profile_vals = []
    ai_profile_vals  = []
    for cl in classes:
        lab_vals = [math.log2(max(r["lab_mic"],0.03)) for r in sim["mic_results"] if r["class"]==cl]
        ai_vals  = [math.log2(max(r["ai_mic"], 0.03)) for r in sim["mic_results"] if r["class"]==cl]
        lab_profile_vals.append(round(sum(lab_vals)/len(lab_vals), 2) if lab_vals else 0)
        ai_profile_vals.append(round(sum(ai_vals)/len(ai_vals),  2) if ai_vals  else 0)

    # ── MRI comparison ──
    mri_delta = round(abs(sim["lab_mri"] - sim["ai_mri"]), 3)
    ari_delta = round(abs(sim["lab_ari"] - sim["ai_ari"]), 3)
    mri_match = "IDENTICAL" if mri_delta < 0.03 else "NEAR MATCH" if mri_delta < 0.08 else "DIVERGED"
    risk_agree = sim["lab_risk"][0] == sim["ai_risk"][0]

    # ── MIC rows HTML ──
    mic_rows = ""
    for r in sim["mic_results"]:
        lab_cls = {"S":"#3B6D11","I":"#BA7517","R":"#A32D2D"}[r["lab_interp"]]
        ai_cls  = {"S":"#3B6D11","I":"#BA7517","R":"#A32D2D"}[r["ai_interp"]]
        match   = r["lab_interp"] == r["ai_interp"]
        match_sym = "✓" if match else "≈" if abs(r["lab_mic"]-r["ai_mic"]) < r["S_break"]*2 else "✗"
        match_col = "#3B6D11" if match else "#BA7517" if match_sym=="≈" else "#A32D2D"
        mic_rows += f"""
        <tr>
          <td style="padding:8px 10px;font-size:13px;">{r['name']}</td>
          <td style="padding:8px 10px;font-size:11px;color:#888;">{r['class']}</td>
          <td style="padding:8px 10px;font-size:13px;font-weight:500;">{r['lab_mic']} µg/mL</td>
          <td style="padding:8px 10px;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{lab_cls}22;color:{lab_cls};">{r['lab_interp']}</span></td>
          <td style="padding:8px 10px;font-size:13px;font-weight:500;">{r['ai_mic']} µg/mL</td>
          <td style="padding:8px 10px;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{ai_cls}22;color:{ai_cls};">{r['ai_interp']}</span></td>
          <td style="padding:8px 10px;font-size:18px;color:{match_col};text-align:center;">{match_sym}</td>
        </tr>"""

    # ── colonies SVG ──
    rng2 = random.Random(hash(sim["bacteria"]) % 9999)
    colony_svg = ""
    for _ in range(sim["colony_count"]):
        cx = rng2.randint(15, 185)
        cy = rng2.randint(15, 185)
        r2 = rng2.randint(3, 9)
        colony_svg += f'<circle cx="{cx}" cy="{cy}" r="{r2}" fill="{colony_c}" stroke="#aaa" stroke-width="0.5" opacity="0.85"/>'
    if db["shape"] == "swarming":
        colony_svg += '<text x="100" y="195" text-anchor="middle" font-size="9" fill="#888">Swarming visible</text>'

    # ── gram microscopy SVG ──
    gram_fill = gram_col
    gram_shape_svg = ""
    shape = db["shape"].lower()
    gram_rng = random.Random(42)
    if "coccus" in shape or "cocci" in shape:
        for _ in range(18):
            gx = gram_rng.randint(10, 110)
            gy = gram_rng.randint(10, 70)
            gram_shape_svg += f'<ellipse cx="{gx}" cy="{gy}" rx="5" ry="5" fill="{gram_fill}" opacity="0.8"/>'
            if db["arrangement"] in ["Diplococci", "Pairs/Chains", "Diplococci/Chains"]:
                gram_shape_svg += f'<ellipse cx="{gx+11}" cy="{gy}" rx="5" ry="5" fill="{gram_fill}" opacity="0.8"/>'
    else:
        for _ in range(14):
            gx = gram_rng.randint(5, 90)
            gy = gram_rng.randint(5, 65)
            ang = gram_rng.randint(0, 180)
            gram_shape_svg += f'<rect x="{gx}" y="{gy}" width="18" height="7" rx="3" fill="{gram_fill}" opacity="0.8" transform="rotate({ang},{gx+9},{gy+3})"/>'

    # ── summary stats for match section ──
    matched_ab  = sum(1 for r in sim["mic_results"] if r["lab_interp"] == r["ai_interp"])
    total_ab    = len(sim["mic_results"])
    match_pct   = round(matched_ab / total_ab * 100)

    # ── risk badge colors ──
    def risk_badge(risk_tuple):
        label, col, icon = risk_tuple
        return f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;background:{col}22;color:{col};font-size:12px;font-weight:700;">{icon} {label}</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Virtual Lab — {sim['bacteria']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Sora',sans-serif;background:#0a0f1a;color:#e2e8f0;min-height:100vh}}
  :root{{
    --accent:#00c9ff;--accent2:#7c3aed;--success:#22c55e;--warn:#f59e0b;--danger:#ef4444;
    --card:#111827;--border:rgba(255,255,255,0.07);--muted:#64748b;
  }}
  h1,h2,h3{{color:#f1f5f9}}

  .lab-shell{{max-width:1200px;margin:0 auto;padding:24px 20px}}

  /* TOP HEADER */
  .lab-header{{
    background:linear-gradient(135deg,#0c1f3f 0%,#111827 60%,#0f1a2e 100%);
    border:1px solid var(--border);border-radius:16px;padding:24px 28px;
    margin-bottom:20px;position:relative;overflow:hidden;
  }}
  .lab-header::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2),#10b981,var(--accent));
    background-size:300% auto;animation:scanBar 4s linear infinite}}
  @keyframes scanBar{{0%{{background-position:0% center}}100%{{background-position:300% center}}}}
  .header-grid{{display:grid;grid-template-columns:1fr auto;gap:20px;align-items:start}}
  .bac-name{{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:600;color:#00c9ff;margin-bottom:4px}}
  .bac-sub{{font-size:13px;color:var(--muted)}}
  .header-badges{{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;margin-top:6px}}

  /* TABS */
  .tab-nav{{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:20px;overflow-x:auto}}
  .tab-btn{{
    padding:10px 18px;font-size:13px;font-weight:600;color:var(--muted);
    background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;
    white-space:nowrap;transition:all 0.2s;font-family:'Sora',sans-serif
  }}
  .tab-btn:hover{{color:#e2e8f0}}
  .tab-btn.active{{color:var(--accent);border-bottom-color:var(--accent)}}
  .tab-panel{{display:none}}.tab-panel.active{{display:block}}

  /* CARDS */
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin-bottom:16px}}
  .card-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);margin-bottom:14px}}
  .grid-2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
  .grid-3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
  .grid-4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
  .grid-5{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}}

  /* METRIC */
  .metric{{background:#0d1420;border:1px solid var(--border);border-radius:10px;padding:14px 16px;text-align:center}}
  .metric-lbl{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:6px}}
  .metric-val{{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace}}
  .metric-sub{{font-size:10px;color:var(--muted);margin-top:3px}}

  /* MATCH CARD */
  .match-card{{border-radius:10px;padding:12px 16px;margin-bottom:10px;display:flex;align-items:center;gap:14px}}
  .match-icon{{font-size:20px;flex-shrink:0}}
  .match-label{{font-size:13px;font-weight:600}}
  .match-detail{{font-size:11px;color:var(--muted);margin-top:2px}}
  .match-ok{{background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2)}}
  .match-near{{background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2)}}
  .match-diff{{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2)}}

  /* MIC TABLE */
  .mic-table{{width:100%;border-collapse:collapse}}
  .mic-table th{{padding:8px 10px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);border-bottom:1px solid var(--border);text-align:left;font-weight:600}}
  .mic-table td{{border-bottom:1px solid rgba(255,255,255,0.03)}}
  .mic-table tr:last-child td{{border-bottom:none}}
  .mic-table tr:hover td{{background:rgba(255,255,255,0.02)}}

  /* CHART WRAPPERS */
  .chart-wrap{{position:relative;width:100%;height:220px}}
  .chart-wrap-lg{{position:relative;width:100%;height:280px}}
  .chart-wrap-xl{{position:relative;width:100%;height:320px}}

  /* MICROSCOPE */
  .scope-viewport{{width:200px;height:200px;border-radius:50%;border:4px solid #334155;overflow:hidden;background:#f5f0e8;margin:0 auto}}
  .scope-viewport svg{{width:100%;height:100%}}

  /* PLATE */
  .plate-viewport{{width:200px;height:200px;border-radius:50%;border:4px solid #334155;background:#1a2e1a;margin:0 auto;position:relative;overflow:hidden}}
  .plate-label{{text-align:center;font-size:12px;color:var(--muted);margin-top:8px}}

  /* BADGE */
  .badge{{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700}}
  .badge-s{{background:rgba(34,197,94,0.15);color:#22c55e}}
  .badge-i{{background:rgba(245,158,11,0.15);color:#f59e0b}}
  .badge-r{{background:rgba(239,68,68,0.15);color:#ef4444}}

  /* GENE TAGS */
  .gene-tag{{display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;font-family:'JetBrains Mono',monospace;background:rgba(0,201,255,0.08);border:1px solid rgba(0,201,255,0.2);color:#67e8f9;margin:2px}}

  /* PROGRESS BAR */
  .prog-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
  .prog-label{{font-size:12px;min-width:120px;color:#e2e8f0}}
  .prog-bar{{flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden}}
  .prog-fill{{height:100%;border-radius:3px;transition:width .6s ease}}
  .prog-val{{font-size:11px;min-width:36px;text-align:right;color:var(--muted)}}

  /* ALERT BAR */
  .alert{{padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px}}
  .alert-high{{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#f87171}}
  .alert-mod{{background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);color:#fbbf24}}
  .alert-low{{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:#4ade80}}

  /* DIVIDER */
  hr{{border:none;border-top:1px solid var(--border);margin:16px 0}}

  /* SCAN ANIMATION */
  @keyframes fadeUp{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
  .animate{{animation:fadeUp .35s ease both}}

  /* INFO ROWS */
  .info-row{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:13px}}
  .info-row:last-child{{border-bottom:none}}
  .info-key{{color:var(--muted)}}
  .info-val{{font-weight:600;text-align:right}}

  /* SUSCEPTIBILITY */
  .susc-zone{{background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:14px 16px}}
  .susc-zone-title{{color:#22c55e;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}}

  /* LOG BOX */
  .log-box{{background:#060d1a;border:1px solid var(--border);border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:11px;max-height:220px;overflow-y:auto;line-height:1.8}}

  @media(max-width:640px){{
    .bac-name{{font-size:1.1rem}}
    .header-grid{{grid-template-columns:1fr}}
  }}
</style>
</head>
<body>
<div class="lab-shell">

<!-- HEADER -->
<div class="lab-header animate">
  <div class="header-grid">
    <div>
      <div class="bac-name">🔬 {sim['bacteria']}</div>
      <div class="bac-sub" style="margin-top:4px">{db['disease']}</div>
      <div class="header-badges" style="margin-top:10px;justify-content:flex-start;">
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

<!-- TABS -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('overview', this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('culture', this)">Culture</button>
  <button class="tab-btn" onclick="switchTab('mic', this)">MIC Plate</button>
  <button class="tab-btn" onclick="switchTab('compare', this)">Lab vs AI</button>
  <button class="tab-btn" onclick="switchTab('profile', this)">Drug Profile</button>
  <button class="tab-btn" onclick="switchTab('genes', this)">Genes & Identity</button>
</div>

<!-- ════════════════════════════════════════════════════════════
     TAB 1: OVERVIEW
════════════════════════════════════════════════════════════ -->
<div id="tab-overview" class="tab-panel active animate">

  {'<div class="alert alert-high">⚠️ HIGH PRIORITY PATHOGEN — MRI Score exceeds 0.35 threshold</div>' if sim["lab_risk"][0]=="HIGH" else
   '<div class="alert alert-mod">⚡ MODERATE RISK — Antibiogram-guided therapy recommended</div>' if sim["lab_risk"][0]=="MODERATE" else
   '<div class="alert alert-low">✓ LOW RISK — Standard empirical therapy protocols viable</div>'}

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
      <div class="info-row"><span class="info-key">Oxidase test</span><span class="info-val">{db['oxidase']}</span></div>
      <div class="info-row"><span class="info-key">Catalase test</span><span class="info-val">{db['catalase']}</span></div>
      <div class="info-row"><span class="info-key">Lactose fermentation</span><span class="info-val">{'Positive' if db['ferments_lactose'] else 'Negative'}</span></div>
      <div class="info-row"><span class="info-key">Selective medium</span><span class="info-val" style="max-width:180px;text-align:right">{db['selective_media']}</span></div>
    </div>
    <div class="card">
      <div class="card-title">Resistance profile summary</div>
      {"".join([f'''<div class="prog-row">
        <span class="prog-label">{cl}</span>
        <div class="prog-bar"><div class="prog-fill" style="width:{round(db['base_resistance'].get(cl,0)*100)}%;background:{'#ef4444' if db['base_resistance'].get(cl,0)>0.5 else '#f59e0b' if db['base_resistance'].get(cl,0)>0.25 else '#22c55e'};"></div></div>
        <span class="prog-val">{round(db['base_resistance'].get(cl,0)*100)}%</span>
      </div>''' for cl in ["Beta-lactam","Fluoroquinolone","Aminoglycoside","Carbapenem","Tetracycline","Macrolide","Glycopeptide","Colistin"]])}
      <div style="font-size:10px;color:var(--muted);margin-top:10px">* Based on population-level reference benchmarks</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">AI vs Lab match summary</div>
    <div class="grid-3">
      <div class="match-card {'match-ok' if mri_delta < 0.03 else 'match-near' if mri_delta < 0.08 else 'match-diff'}">
        <span class="match-icon">{'✓' if mri_delta<0.03 else '~' if mri_delta<0.08 else '✗'}</span>
        <div><div class="match-label">MRI reading</div><div class="match-detail">Lab: {sim['lab_mri']} | AI: {sim['ai_mri']} | Δ={mri_delta}</div></div>
      </div>
      <div class="match-card {'match-ok' if ari_delta < 0.02 else 'match-near' if ari_delta < 0.06 else 'match-diff'}">
        <span class="match-icon">{'✓' if ari_delta<0.02 else '~' if ari_delta<0.06 else '✗'}</span>
        <div><div class="match-label">ARI reading</div><div class="match-detail">Lab: {sim['lab_ari']} | AI: {sim['ai_ari']} | Δ={ari_delta}</div></div>
      </div>
      <div class="match-card {'match-ok' if risk_agree else 'match-diff'}">
        <span class="match-icon">{'✓' if risk_agree else '✗'}</span>
        <div><div class="match-label">Risk classification</div><div class="match-detail">Lab: {risk_lab[0]} | AI: {risk_ai[0]}</div></div>
      </div>
    </div>
    <div style="margin-top:10px;padding:10px 14px;background:rgba(0,201,255,0.05);border-radius:8px;font-size:12px;color:#94a3b8">
      Antibiotic concordance: <strong style="color:{'#22c55e' if match_pct>=80 else '#f59e0b' if match_pct>=60 else '#ef4444'}">{matched_ab}/{total_ab} agents ({match_pct}%)</strong> agree between lab and AI interpretations.
      {f'<span style="color:#22c55e">  High concordance — AI model well-calibrated for this organism.</span>' if match_pct >= 80 else f'<span style="color:#f59e0b">  Moderate concordance — some discrepancy, review individual MICs.</span>' if match_pct >= 60 else '<span style="color:#ef4444">  Low concordance — recommend phenotypic confirmation.</span>'}
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════
     TAB 2: CULTURE
════════════════════════════════════════════════════════════ -->
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
      <div class="chart-wrap">
        <canvas id="growthChart" role="img" aria-label="Bacterial growth curve over incubation time showing CFU per mL">Growth data loading.</canvas>
      </div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px">Medium: {sim['medium']} · Inoculum: 10<sup>{sim['cfu_exp']}</sup> CFU/mL · {sim['incubation_h']}h total</div>
    </div>
    <div class="card">
      <div class="card-title">Colony morphology plate</div>
      <div class="plate-viewport">
        <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">{colony_svg}</svg>
      </div>
      <div class="plate-label">{db['colony_size'].capitalize()} colonies · {db['selective_media']}</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Gram stain microscopy (×1000 oil immersion)</div>
    <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">
      <div>
        <div class="scope-viewport">
          <svg width="120" height="80" xmlns="http://www.w3.org/2000/svg" style="background:#f5f0e8">
            {gram_shape_svg}
          </svg>
        </div>
        <div class="plate-label" style="margin-top:8px">Gram {db['gram']} · {db['shape']}</div>
      </div>
      <div style="flex:1;min-width:200px">
        <div class="info-row"><span class="info-key">Gram stain result</span><span class="info-val" style="color:{db['gram_color']}">{'Purple (crystal violet retained)' if db['gram']=='Positive' else 'Pink/red (safranin stained)' if db['gram']=='Negative' else 'Red (acid-fast stain)'}</span></div>
        <div class="info-row"><span class="info-key">Cell morphology</span><span class="info-val">{db['shape']}</span></div>
        <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
        <div class="info-row"><span class="info-key">Spore formation</span><span class="info-val">{'Endospores visible' if db['spore'] else 'None'}</span></div>
        <div class="info-row"><span class="info-key">Capsule</span><span class="info-val">{'Present (India ink halo)' if db['capsule'] else 'Absent'}</span></div>
        <div class="info-row"><span class="info-key">Motility test</span><span class="info-val">{'Positive (turbid growth)' if db['motility'] else 'Negative (along stab line)'}</span></div>
        <div class="info-row"><span class="info-key">Oxidase</span><span class="info-val">{db['oxidase']}</span></div>
        <div class="info-row"><span class="info-key">Catalase</span><span class="info-val">{db['catalase']}</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════
     TAB 3: MIC PLATE
════════════════════════════════════════════════════════════ -->
<div id="tab-mic" class="tab-panel animate">

  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:14px">
    <span style="font-size:12px;color:var(--muted)">Select antibiotic to view plate:</span>
    <div id="ab-selector" style="display:flex;gap:6px;flex-wrap:wrap"></div>
  </div>

  <div class="card" id="mic-plate-card">
    <div class="card-title" id="mic-plate-title">MIC Plate</div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:12px">
      96-well format · Rows = replicates · Columns = doubling dilutions · Blue ring = MIC breakpoint
    </div>
    <div id="mic-plate-grid" style="display:grid;grid-template-columns:repeat(12,1fr);gap:4px;margin-bottom:8px"></div>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--muted);padding:0 2px">
      <span>0.06</span><span>0.12</span><span>0.25</span><span>0.5</span><span>1</span><span>2</span><span>4</span><span>8</span><span>16</span><span>32</span><span>64</span><span>128 µg/mL</span>
    </div>
    <div style="margin-top:14px;padding:10px 16px;background:#0d1420;border-radius:8px;font-size:13px">
      MIC = <strong id="mic-result-val" style="color:#00c9ff;font-family:'JetBrains Mono',monospace">—</strong> µg/mL &nbsp;|&nbsp;
      Interpretation: <strong id="mic-result-interp">—</strong>
      &nbsp;|&nbsp; S breakpoint ≤ <span id="mic-s-break">—</span> µg/mL &nbsp;|&nbsp; R breakpoint > <span id="mic-r-break">—</span> µg/mL
    </div>
    <div style="display:flex;gap:14px;margin-top:10px;font-size:11px;flex-wrap:wrap">
      <span style="display:flex;align-items:center;gap:4px"><span style="width:12px;height:12px;border-radius:50%;background:#ef4444;display:inline-block"></span>Growth (resistant)</span>
      <span style="display:flex;align-items:center;gap:4px"><span style="width:12px;height:12px;border-radius:50%;background:#22c55e;display:inline-block"></span>No growth (inhibited)</span>
      <span style="display:flex;align-items:center;gap:4px"><span style="width:12px;height:12px;border-radius:50%;background:#1d4ed8;border:2px solid #60a5fa;display:inline-block"></span>MIC point</span>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Complete MIC summary — all antibiotics</div>
    <div style="overflow-x:auto">
      <table class="mic-table">
        <thead>
          <tr>
            <th>Antibiotic</th><th>Drug class</th>
            <th>Lab MIC</th><th>Lab result</th>
            <th>AI MIC</th><th>AI result</th>
            <th style="text-align:center">Match</th>
          </tr>
        </thead>
        <tbody>{mic_rows}</tbody>
      </table>
    </div>
    <div style="display:flex;gap:16px;margin-top:12px;font-size:12px;flex-wrap:wrap">
      <span style="color:#22c55e">S = Susceptible</span>
      <span style="color:#f59e0b">I = Intermediate</span>
      <span style="color:#ef4444">R = Resistant</span>
      <span style="color:var(--muted)">✓ = Same class &nbsp; ≈ = Near &nbsp; ✗ = Diverged</span>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════
     TAB 4: LAB vs AI COMPARISON
════════════════════════════════════════════════════════════ -->
<div id="tab-compare" class="tab-panel animate">

  <div class="grid-4" style="margin-bottom:14px">
    <div class="metric">
      <div class="metric-lbl">Lab MRI</div>
      <div class="metric-val" style="color:{risk_lab[1]}">{sim['lab_mri']}</div>
      <div class="metric-sub">{risk_badge(risk_lab)}</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">AI MRI</div>
      <div class="metric-val" style="color:{risk_ai[1]}">{sim['ai_mri']}</div>
      <div class="metric-sub">{risk_badge(risk_ai)}</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">MRI Δ</div>
      <div class="metric-val" style="color:{'#22c55e' if mri_delta<0.03 else '#f59e0b' if mri_delta<0.08 else '#ef4444'}">{mri_delta}</div>
      <div class="metric-sub">{mri_match}</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">AB Concordance</div>
      <div class="metric-val" style="color:{'#22c55e' if match_pct>=80 else '#f59e0b' if match_pct>=60 else '#ef4444'}">{match_pct}%</div>
      <div class="metric-sub">{matched_ab}/{total_ab} agents</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">MRI / ARI bar comparison</div>
      <div class="chart-wrap">
        <canvas id="mriChart" role="img" aria-label="Bar chart comparing lab vs AI MRI and ARI scores">MRI comparison data.</canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Risk classification confidence</div>
      <div id="risk-conf" style="padding:8px 0"></div>
      <hr style="margin:12px 0">
      <div class="card-title">Reading agreement log</div>
      <div class="log-box" id="agreement-log"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">MIC comparison — Lab vs AI (log₂ scale)</div>
    <div class="chart-wrap-xl">
      <canvas id="micCompChart" role="img" aria-label="Grouped bar chart of MIC values comparing lab and AI for each antibiotic">MIC comparison data.</canvas>
    </div>
    <div style="display:flex;gap:16px;margin-top:10px;font-size:11px">
      <span style="display:flex;align-items:center;gap:4px"><span style="width:12px;height:6px;background:#185FA5;display:inline-block;border-radius:2px"></span>Lab reading</span>
      <span style="display:flex;align-items:center;gap:4px"><span style="width:12px;height:6px;background:rgba(0,201,255,0.4);border:1px solid #00c9ff;display:inline-block;border-radius:2px"></span>AI prediction</span>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════
     TAB 5: DRUG PROFILE
════════════════════════════════════════════════════════════ -->
<div id="tab-profile" class="tab-panel animate">

  <div class="card">
    <div class="card-title">Drug class resistance profile — Lab vs AI</div>
    <div class="chart-wrap-xl">
      <canvas id="profileChart" role="img" aria-label="Radar chart comparing lab and AI drug class resistance profiles">Drug profile data.</canvas>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Susceptibility zone analysis</div>
    <div class="susc-zone" style="margin-bottom:12px">
      <div class="susc-zone-title">✓ Potentially susceptible drug classes (zero/low resistance markers)</div>
      <div id="susc-list" style="display:flex;flex-wrap:wrap;gap:6px"></div>
    </div>
    <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:14px 16px">
      <div style="color:#f87171;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">✗ Resistant drug classes (elevated markers)</div>
      <div id="resist-list" style="display:flex;flex-wrap:wrap;gap:6px"></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:10px">
      ⚠️ Clinical confirmation via MIC testing (antibiogram) required before therapeutic application.
    </div>
  </div>

  <div class="card">
    <div class="card-title">Resistance intensity — all drug classes</div>
    {"".join([f'''<div class="prog-row">
      <span class="prog-label" style="font-size:11px">{ab["class"] if i<len(ANTIBIOTICS) and ab["class"] not in [ANTIBIOTICS[j]["class"] for j in range(i)] else ""}</span>
      <div class="prog-bar"><div class="prog-fill" style="width:{round(min(r["lab_mic"]/r["R_break"],1)*100)}%;background:{'#ef4444' if r["lab_interp"]=="R" else '#f59e0b' if r["lab_interp"]=="I" else '#22c55e'}"></div></div>
      <span style="font-size:11px;min-width:80px;text-align:right;color:{'#ef4444' if r['lab_interp']=='R' else '#f59e0b' if r['lab_interp']=='I' else '#22c55e'}">{r["name"]}: {r["lab_mic"]} µg/mL</span>
    </div>''' for i,(ab,r) in enumerate(zip(ANTIBIOTICS, sim["mic_results"]))])}
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════
     TAB 6: GENES & IDENTITY
════════════════════════════════════════════════════════════ -->
<div id="tab-genes" class="tab-panel animate">
  <div class="grid-3" style="margin-bottom:14px">
    <div class="metric"><div class="metric-lbl">Total ARGs</div><div class="metric-val" style="color:#00c9ff">{sim['real_genes']}</div><div class="metric-sub">from CARD JSON</div></div>
    <div class="metric"><div class="metric-lbl">Drug classes</div><div class="metric-val">{sim['real_u_drugs']}</div><div class="metric-sub">unique classes</div></div>
    <div class="metric"><div class="metric-lbl">Mechanisms</div><div class="metric-val">{sim['real_u_mechs']}</div><div class="metric-sub">unique strategies</div></div>
  </div>

  <div class="card">
    <div class="card-title">Known resistance determinants for this species</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px">
      {"".join(f'<span class="gene-tag">{g}</span>' for g in db['notable_genes'])}
    </div>
    <div style="font-size:11px;color:var(--muted)">* Curated reference list. Actual ARGs detected are in your CARD JSON file.</div>
  </div>

  <div class="card">
    <div class="card-title">Biochemical identity panel</div>
    <div class="grid-2">
      <div>
        <div class="info-row"><span class="info-key">Gram stain</span><span class="info-val">{db['gram']}</span></div>
        <div class="info-row"><span class="info-key">Shape</span><span class="info-val">{db['shape']}</span></div>
        <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
        <div class="info-row"><span class="info-key">Motility</span><span class="info-val">{'Yes' if db['motility'] else 'No'}</span></div>
        <div class="info-row"><span class="info-key">Spore formation</span><span class="info-val">{'Yes' if db['spore'] else 'No'}</span></div>
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
  lab_mri:      {sim['lab_mri']},
  ai_mri:       {sim['ai_mri']},
  lab_ari:      {sim['lab_ari']},
  ai_ari:       {sim['ai_ari']},
}};

/* ── TABS ── */
function switchTab(name, btn) {{
  document.querySelectorAll('.tab-panel').forEach(function(p){{p.classList.remove('active');}});
  document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}}

/* ── GROWTH CHART ── */
new Chart(document.getElementById('growthChart'), {{
  type:'line',
  data:{{
    labels: SIM_DATA.growth_times,
    datasets:[{{
      label:'CFU/mL (×10⁶)',
      data: SIM_DATA.growth_cfu,
      borderColor:'#00c9ff', backgroundColor:'rgba(0,201,255,0.06)',
      fill:true, tension:0.4, pointRadius:2, pointBackgroundColor:'#00c9ff',
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b',font:{{size:10}}}}, grid:{{color:'rgba(255,255,255,0.04)'}}}},
      y:{{ticks:{{color:'#64748b',font:{{size:10}},callback:function(v){{return v+'M';}}}}, grid:{{color:'rgba(255,255,255,0.04)'}},min:0}}
    }}
  }}
}});

/* ── MRI BAR CHART ── */
new Chart(document.getElementById('mriChart'), {{
  type:'bar',
  data:{{
    labels:['MRI Score','ARI Score'],
    datasets:[
      {{label:'Lab reading', data:[SIM_DATA.lab_mri, SIM_DATA.lab_ari], backgroundColor:'#185FA5', borderRadius:4}},
      {{label:'AI prediction', data:[SIM_DATA.ai_mri,  SIM_DATA.ai_ari],  backgroundColor:'rgba(0,201,255,0.3)', borderColor:'#00c9ff', borderWidth:2, borderRadius:4}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b'}}, grid:{{color:'rgba(255,255,255,0.04)'}}}},
      y:{{min:0, max:1, ticks:{{color:'#64748b'}}, grid:{{color:'rgba(255,255,255,0.04)'}}}},
    }}
  }}
}});

/* ── MIC COMPARISON CHART ── */
new Chart(document.getElementById('micCompChart'), {{
  type:'bar',
  data:{{
    labels: SIM_DATA.ab_names,
    datasets:[
      {{label:'Lab MIC (log₂)',   data:SIM_DATA.lab_mics, backgroundColor:'#185FA5', borderRadius:3}},
      {{label:'AI MIC (log₂)', data:SIM_DATA.ai_mics,  backgroundColor:'rgba(0,201,255,0.3)', borderColor:'#00c9ff', borderWidth:1, borderRadius:3}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b',font:{{size:10}}, maxRotation:45}}, grid:{{color:'rgba(255,255,255,0.04)'}}}},
      y:{{ticks:{{color:'#64748b',callback:function(v){{return '2^'+v;}}}}, grid:{{color:'rgba(255,255,255,0.04)'}}, title:{{display:true,text:'log₂(MIC)',color:'#64748b',font:{{size:10}}}}}},
    }}
  }}
}});

/* ── PROFILE RADAR CHART ── */
new Chart(document.getElementById('profileChart'), {{
  type:'radar',
  data:{{
    labels: SIM_DATA.classes,
    datasets:[
      {{label:'Lab reading', data:SIM_DATA.lab_profile, borderColor:'#185FA5', backgroundColor:'rgba(24,95,165,0.15)', pointBackgroundColor:'#185FA5', borderWidth:2}},
      {{label:'AI prediction', data:SIM_DATA.ai_profile, borderColor:'#00c9ff', backgroundColor:'rgba(0,201,255,0.08)', pointBackgroundColor:'#00c9ff', borderWidth:2, borderDash:[5,3]}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{r:{{
      ticks:{{color:'#64748b', backdropColor:'transparent'}},
      grid:{{color:'rgba(255,255,255,0.06)'}},
      pointLabels:{{color:'#94a3b8', font:{{size:11}}}},
    }}}}
  }}
}});

/* ── MIC PLATE ── */
var DILUTIONS_ARR = [0.06,0.12,0.25,0.5,1,2,4,8,16,32,64,128];
var currentMicIdx = 0;

function buildAbSelector() {{
  var sel = document.getElementById('ab-selector');
  SIM_DATA.mic_results.forEach(function(ab, i) {{
    var btn = document.createElement('button');
    btn.textContent = ab.name;
    btn.style.cssText = 'padding:4px 10px;font-size:11px;border-radius:4px;cursor:pointer;border:1px solid rgba(255,255,255,0.12);background:'+(i===0?'rgba(0,201,255,0.15)':'rgba(255,255,255,0.04)')+';color:'+(i===0?'#00c9ff':'#94a3b8')+';font-family:Sora,sans-serif;';
    btn.onclick = function() {{
      currentMicIdx = i;
      document.querySelectorAll('#ab-selector button').forEach(function(b, j) {{
        b.style.background = j===i ? 'rgba(0,201,255,0.15)' : 'rgba(255,255,255,0.04)';
        b.style.color = j===i ? '#00c9ff' : '#94a3b8';
      }});
      renderMicPlate(i);
    }};
    sel.appendChild(btn);
  }});
}}

function renderMicPlate(idx) {{
  var ab  = SIM_DATA.mic_results[idx];
  var mic = ab.lab_mic;
  var micColIdx = DILUTIONS_ARR.indexOf(DILUTIONS_ARR.reduce(function(a,b){{return Math.abs(b-mic)<Math.abs(a-mic)?b:a;}}));
  var grid = document.getElementById('mic-plate-grid');
  grid.innerHTML = '';
  for (var row = 0; row < 8; row++) {{
    for (var col = 0; col < 12; col++) {{
      var well = document.createElement('div');
      well.style.cssText = 'aspect-ratio:1;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;cursor:default;transition:transform .1s;';
      if (col === micColIdx) {{
        well.style.background = '#1d4ed8'; well.style.border = '2px solid #60a5fa'; well.style.color='#fff';
        well.textContent = 'MIC';
      }} else if (col < micColIdx) {{
        well.style.background = 'rgba(239,68,68,0.6)'; well.style.border = '1px solid #ef4444'; well.style.color='#fff';
        well.textContent = '+';
      }} else {{
        well.style.background = 'rgba(34,197,94,0.2)'; well.style.border = '1px solid rgba(34,197,94,0.4)'; well.style.color='#4ade80';
        well.textContent = '−';
      }}
      grid.appendChild(well);
    }}
  }}
  document.getElementById('mic-result-val').textContent = mic;
  var interpEl = document.getElementById('mic-result-interp');
  var iC = {{'S':'#22c55e','I':'#f59e0b','R':'#ef4444'}}[ab.lab_interp];
  interpEl.innerHTML = '<span style="color:'+iC+';font-weight:700;">'+{{'S':'Susceptible','I':'Intermediate','R':'Resistant'}}[ab.lab_interp]+'</span>';
  document.getElementById('mic-s-break').textContent = ab.S_break;
  document.getElementById('mic-r-break').textContent = ab.R_break;
  document.getElementById('mic-plate-title').textContent = ab.name + ' (' + ab.class + ')';
}}

/* ── RISK CONFIDENCE BARS ── */
function buildRiskConf() {{
  var risks = ['LOW','MODERATE','HIGH'];
  var labRisk = '{risk_lab[0]}'; var aiRisk = '{risk_ai[0]}';
  var labMRI = SIM_DATA.lab_mri; var aiMRI = SIM_DATA.ai_mri;
  var conf = {{}};
  conf['LOW']      = labMRI < 0.15 ? 0.70 + Math.random()*0.25 : 0.05 + Math.random()*0.10;
  conf['MODERATE'] = (labMRI>=0.15&&labMRI<0.35) ? 0.60+Math.random()*0.30 : 0.05+Math.random()*0.12;
  conf['HIGH']     = labMRI >= 0.35 ? 0.65+Math.random()*0.30 : 0.03+Math.random()*0.08;
  var total = Object.values(conf).reduce(function(a,b){{return a+b;}},0);
  var el = document.getElementById('risk-conf');
  el.innerHTML = '';
  risks.forEach(function(r) {{
    var pct = Math.round(conf[r]/total*100);
    var col = r==='HIGH'?'#ef4444':r==='MODERATE'?'#f59e0b':'#22c55e';
    el.innerHTML += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">' +
      '<div style="font-size:11px;font-weight:700;min-width:70px;color:'+col+';">'+r+'</div>' +
      '<div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">' +
        '<div style="height:100%;width:'+pct+'%;background:'+col+';border-radius:3px;transition:width .6s ease;"></div></div>' +
      '<div style="font-size:11px;min-width:32px;color:#64748b;">'+pct+'%</div>' +
      (r===labRisk?'<span style="background:'+col+'22;color:'+col+';font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid '+col+'44;">Lab</span>':'') +
      (r===aiRisk?'<span style="background:rgba(0,201,255,0.1);color:#00c9ff;font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid rgba(0,201,255,0.3);">AI</span>':'') +
      '</div>';
  }});
}}

/* ── AGREEMENT LOG ── */
function buildAgreementLog() {{
  var log = document.getElementById('agreement-log');
  var lines = [];
  var ts = function() {{ return new Date().toLocaleTimeString(); }};
  var labMRI = SIM_DATA.lab_mri, aiMRI = SIM_DATA.ai_mri;
  var delta = Math.abs(labMRI - aiMRI);
  lines.push({{t:'ok',  m:'[LAB] MRI computed: ' + labMRI + ' ({risk_lab[0]})'}});
  lines.push({{t:'info',m:'[AI]  MRI predicted: ' + aiMRI + ' ({risk_ai[0]})'}});
  lines.push({{t: delta<0.03?'ok':delta<0.08?'warn':'err', m:'[CMP] MRI delta: ' + delta.toFixed(3) + ' — ' + (delta<0.03?'IDENTICAL':delta<0.08?'NEAR MATCH':'DIVERGED')}});
  lines.push({{t:'ok',  m:'[LAB] ARI computed: ' + SIM_DATA.lab_ari}});
  lines.push({{t:'info',m:'[AI]  ARI predicted: ' + SIM_DATA.ai_ari}});
  var matched = SIM_DATA.mic_results.filter(function(r){{return r.lab_interp===r.ai_interp;}}).length;
  lines.push({{t:matched>=Math.round(SIM_DATA.mic_results.length*0.8)?'ok':'warn', m:'[MIC] Concordance: '+matched+'/'+SIM_DATA.mic_results.length+' agents agree ('+Math.round(matched/SIM_DATA.mic_results.length*100)+'%)'}});
  SIM_DATA.mic_results.forEach(function(r) {{
    var m = r.lab_interp===r.ai_interp;
    lines.push({{t:m?'ok':'warn', m:'[MIC] '+r.name+': Lab='+r.lab_mic+'('+r.lab_interp+') AI='+r.ai_mic+'('+r.ai_interp+') '+(m?'✓':'≈')}});
  }});
  log.innerHTML = lines.map(function(l){{
    var c = l.t==='ok'?'#22c55e':l.t==='info'?'#00c9ff':l.t==='warn'?'#f59e0b':'#ef4444';
    return '<div style="color:'+c+'">'+l.m+'</div>';
  }}).join('');
  log.scrollTop = log.scrollHeight;
}}

/* ── SUSCEPTIBILITY ZONE ── */
function buildSuscZone() {{
  var s_list = document.getElementById('susc-list');
  var r_list = document.getElementById('resist-list');
  SIM_DATA.mic_results.forEach(function(r) {{
    var tag = document.createElement('span');
    tag.style.cssText = 'display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;margin:2px;';
    tag.textContent = r.name;
    if (r.lab_interp === 'S') {{
      tag.style.background = 'rgba(34,197,94,0.1)'; tag.style.color = '#4ade80'; tag.style.border = '1px solid rgba(34,197,94,0.25)';
      s_list.appendChild(tag);
    }} else if (r.lab_interp === 'R') {{
      tag.style.background = 'rgba(239,68,68,0.1)'; tag.style.color = '#f87171'; tag.style.border = '1px solid rgba(239,68,68,0.25)';
      r_list.appendChild(tag);
    }}
  }});
  if (!s_list.children.length) s_list.innerHTML = '<span style="color:#64748b;font-size:12px;">No susceptible agents found in panel</span>';
  if (!r_list.children.length) r_list.innerHTML = '<span style="color:#64748b;font-size:12px;">No fully resistant agents in panel</span>';
}}

/* ── INIT ── */
buildAbSelector();
renderMicPlate(0);
buildRiskConf();
buildAgreementLog();
buildSuscZone();
</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT RENDER FUNCTION  — call this from your tab10
# ─────────────────────────────────────────────────────────────────────────────
def render_lab_tab():
    # ── CSS header for the Streamlit-level controls ──
    st.markdown("""
    <style>
    .lab-section-title{
      font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
      color:#00d4ff;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid rgba(0,212,255,.15);
    }
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
        Full phenotypic simulation for every bacterium in your database — culture, MIC plate, AI vs lab comparison, drug profiles and gene identity.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Scan JSON files ──
    json_files  = [f for f in os.listdir('.') if f.endswith('.json')]
    file_to_bac = scan_json_files()   # {filename: bacteria_name}

    if not json_files:
        st.warning("No JSON files found in the working directory. Upload CARD-format JSON files to begin.")
        return

    # ── Sidebar-like controls in columns ──
    col_bac, col_file = st.columns([2, 2])

    with col_bac:
        st.markdown('<div class="lab-section-title">Select Organism</div>', unsafe_allow_html=True)
        all_bac_names = sorted(BACTERIA_DB.keys())
        # Prefer bacteria whose JSON was found
        found_bac = sorted(set(file_to_bac.values()))
        default_bac = found_bac[0] if found_bac else all_bac_names[0]
        selected_bac = st.selectbox(
            "Bacterial species", all_bac_names,
            index=all_bac_names.index(default_bac),
            label_visibility="collapsed"
        )

    with col_file:
        st.markdown('<div class="lab-section-title">JSON Source File</div>', unsafe_allow_html=True)
        # Files that match this bacterium go first
        matching = [f for f, b in file_to_bac.items() if b == selected_bac]
        file_opts = matching + [f for f in json_files if f not in matching]
        selected_json = st.selectbox("JSON file", file_opts, label_visibility="collapsed")

    st.markdown("---")

    # ── Lab condition sliders ──
    db = BACTERIA_DB[selected_bac]
    st.markdown('<div class="lab-section-title">Lab Conditions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        temperature = st.slider("Temperature (°C)", 25.0, 45.0, float(db["optimal_temp"]), 0.5)
    with c2:
        ph = st.slider("pH", 5.5, 9.0, float(db["optimal_ph"]), 0.1)
    with c3:
        incubation_h = st.slider("Incubation (h)", 4, 72, 18, 1)
    with c4:
        cfu_exp = st.slider("Inoculum 10^x CFU/mL", 3, 8, 5, 1)
    with c5:
        resistance_override = st.slider("Resistance pressure", 0.0, 1.0, 0.5, 0.05,
                                        help="Simulate high-pressure selection environments (e.g. hospital vs environmental)")

    medium = st.selectbox("Growth medium",
        ["Mueller-Hinton Broth (MHB)", "Lysogeny Broth Agar (LBA)",
         "Columbia Blood Agar (CBA)", "Brain Heart Infusion (BHI)",
         "Löwenstein-Jensen (LJ)", "Chocolate Agar"])

    st.markdown("---")

    # ── Presets ──
    preset_col, run_col = st.columns([3, 1])
    with preset_col:
        preset = st.selectbox("Quick presets",
            ["— custom —", "MRSA (High resistance)", "ESBL E. coli", "XDR Acinetobacter",
             "Pan-susceptible Salmonella", "MDR M. tuberculosis", "VRE Enterococcus"],
            label_visibility="visible")

    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

    # ── Apply presets ──
    preset_map = {
        "MRSA (High resistance)":       {"bac": "Staphylococcus aureus",        "res": 0.85},
        "ESBL E. coli":                 {"bac": "Escherichia coli",              "res": 0.72},
        "XDR Acinetobacter":            {"bac": "Acinetobacter baumannii",       "res": 0.92},
        "Pan-susceptible Salmonella":   {"bac": "Salmonella enterica",           "res": 0.10},
        "MDR M. tuberculosis":          {"bac": "Mycobacterium tuberculosis",    "res": 0.78},
        "VRE Enterococcus":             {"bac": "Enterococcus faecium",          "res": 0.80},
    }
    if preset in preset_map:
        resistance_override = preset_map[preset]["res"]
        selected_bac        = preset_map[preset]["bac"]

    # ── Run & Render ──
    if 'lab_result_html' not in st.session_state:
        st.session_state.lab_result_html = None
        st.session_state.lab_run_key    = None

    run_key = f"{selected_bac}_{selected_json}_{temperature}_{ph}_{incubation_h}_{cfu_exp}_{resistance_override}_{medium}"

    if run_btn or (st.session_state.lab_run_key == run_key and st.session_state.lab_result_html):
        if run_btn or st.session_state.lab_result_html is None:
            with st.spinner(f"Running virtual lab for {selected_bac}..."):
                sim  = simulate_lab(
                    selected_bac, selected_json,
                    resistance_override, temperature, ph,
                    incubation_h, cfu_exp, medium
                )
                html = build_lab_html(sim)
                st.session_state.lab_result_html = html
                st.session_state.lab_run_key     = run_key

        components.html(st.session_state.lab_result_html, height=900, scrolling=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:rgba(15,23,42,0.5);border:1px dashed rgba(0,212,255,.2);border-radius:14px;margin-top:10px;">
          <div style="font-size:3rem;margin-bottom:12px;">🧫</div>
          <div style="font-size:18px;color:#e2e8f0;font-weight:600;margin-bottom:8px;">Virtual Lab Ready</div>
          <div style="font-size:13px;color:#64748b;">
            Configure the organism and lab conditions above, then click <strong style="color:#00d4ff;">Run Simulation</strong>.<br>
            The lab will generate full culture, MIC plate, AI comparison, and drug profile visualisations.
          </div>
        </div>
        """, unsafe_allow_html=True)
