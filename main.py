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
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon=" 🧬 ")

st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #ffffff;}

.main-header {
background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
padding: 2.5rem;
border-radius: 15px;
color: white;
text-align: center;
margin-bottom: 2rem;
box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}

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
background-color: #1e293b;
border: 1px solid #334155;
padding: 1.5rem;
border-radius: 12px;
text-align: center;
}
.metric-label {
color: #94a3b8;
font-size: 0.8rem;
font-weight: bold;
text-transform: uppercase;
letter-spacing: 0.05em;
}
.metric-value {
color: #ffffff;
font-size: 1.8rem;
font-weight: 700;
margin-top: 0.5rem;
}

.report-card {
background-color: #1e293b;
border-left: 5px solid #3b82f6;
padding: 20px;
border-radius: 10px;
margin-bottom: 25px;
}
.report-header {
color: #3b82f6;
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
background: rgba(59, 130, 246, 0.1);
color: #60a5fa;
padding: 4px 12px;
border-radius: 20px;
font-size: 0.85rem;
border: 1px solid rgba(59, 130, 246, 0.3);
}

.math-card {
background-color: #1e293b;
border: 1px solid #334155;
padding: 25px;
border-radius: 12px;
margin-bottom: 20px;
}
.math-card-header {
color: #3b82f6;
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
background: rgba(30, 41, 59, 0.8);
border-radius: 12px;
border-left: 5px solid #3b82f6;
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# LANDING PAGE
# ==========================================
if 'show_landing' not in st.session_state:
    st.session_state.show_landing = True

if st.session_state.show_landing:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }

    body, .main, .block-container {
        background: #000 !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    #landing-canvas {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
        background: radial-gradient(ellipse at 50% 30%, #001a2e 0%, #000510 60%, #000 100%);
    }

    .landing-wrap {
        position: relative;
        z-index: 10;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        font-family: 'Rajdhani', sans-serif;
    }

    .logo-ring {
        width: 110px; height: 110px;
        border-radius: 50%;
        border: 2px solid rgba(59,130,246,0.4);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 28px;
        position: relative;
        animation: ringPulse 3s ease-in-out infinite;
        box-shadow: 0 0 40px rgba(59,130,246,0.2), inset 0 0 40px rgba(59,130,246,0.05);
    }
    .logo-ring::before {
        content: '';
        position: absolute;
        inset: -8px;
        border-radius: 50%;
        border: 1px solid rgba(59,130,246,0.15);
        animation: ringPulse 3s ease-in-out infinite 0.5s;
    }
    .logo-ring::after {
        content: '';
        position: absolute;
        inset: -18px;
        border-radius: 50%;
        border: 1px solid rgba(59,130,246,0.07);
        animation: ringPulse 3s ease-in-out infinite 1s;
    }
    .logo-emoji { font-size: 52px; line-height: 1; animation: floatIcon 4s ease-in-out infinite; }

    @keyframes ringPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.04); opacity: 0.85; }
    }
    @keyframes floatIcon {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    .landing-title {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: clamp(2rem, 5vw, 3.6rem);
        text-align: center;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 40%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 12px;
        letter-spacing: 0.04em;
        animation: titleSlide 0.9s cubic-bezier(.22,1,.36,1) both;
    }
    .landing-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: clamp(1rem, 2.5vw, 1.4rem);
        color: #7dd3fc;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        text-align: center;
        margin: 0 0 40px;
        animation: titleSlide 1.1s cubic-bezier(.22,1,.36,1) both 0.1s;
    }
    @keyframes titleSlide {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .landing-divider {
        width: 200px; height: 1px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        margin: 0 auto 48px;
        animation: fadeIn 1.4s ease both 0.3s;
    }
    @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
    }

    .about-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        max-width: 960px;
        width: 100%;
        margin-bottom: 48px;
        animation: fadeIn 1.2s ease both 0.5s;
    }
    .about-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 28px 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .about-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .about-card:hover {
        transform: translateY(-6px);
        border-color: rgba(59,130,246,0.5);
        box-shadow: 0 20px 40px rgba(59,130,246,0.15);
    }
    .about-card:hover::before { opacity: 1; }
    .about-card-icon { font-size: 28px; margin-bottom: 14px; display: block; }
    .about-card-title {
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #60a5fa;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .about-card-text {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.7;
        font-weight: 300;
    }

    .pipeline-wrap {
        max-width: 960px; width: 100%;
        margin-bottom: 48px;
        animation: fadeIn 1.2s ease both 0.7s;
    }
    .section-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.25em;
        color: #3b82f6;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 28px;
    }
    .pipeline {
        display: flex;
        align-items: stretch;
        gap: 0;
        flex-wrap: wrap;
        justify-content: center;
    }
    .pipe-step {
        flex: 1; min-width: 140px; max-width: 180px;
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        position: relative;
        margin: 6px;
        transition: all 0.3s ease;
    }
    .pipe-step:hover { border-color: #3b82f6; background: rgba(30,58,138,0.3); }
    .pipe-num {
        font-family: 'Orbitron', monospace;
        font-size: 1.6rem;
        font-weight: 900;
        color: rgba(59,130,246,0.3);
        line-height: 1;
        margin-bottom: 8px;
    }
    .pipe-step:hover .pipe-num { color: rgba(59,130,246,0.7); }
    .pipe-icon { font-size: 22px; margin-bottom: 8px; display: block; }
    .pipe-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: 0.05em;
        line-height: 1.4;
    }

    .team-wrap {
        max-width: 960px; width: 100%;
        margin-bottom: 52px;
        animation: fadeIn 1.2s ease both 0.9s;
    }
    .team-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: center;
    }
    .team-pill {
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 40px;
        padding: 10px 20px;
        display: flex; align-items: center; gap: 10px;
        transition: all 0.3s ease;
    }
    .team-pill:hover {
        background: rgba(30,58,138,0.5);
        border-color: #60a5fa;
        transform: scale(1.04);
    }
    .team-avatar {
        width: 32px; height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700;
        color: white; letter-spacing: 0.02em;
        flex-shrink: 0;
    }
    .team-name { font-size: 0.9rem; color: #e2e8f0; font-weight: 600; }

    .enter-btn-wrap {
        animation: fadeIn 1.3s ease both 1.1s;
        text-align: center;
        margin-bottom: 30px;
    }
    .enter-btn-sub {
        font-size: 0.75rem;
        color: #475569;
        margin-top: 10px;
        letter-spacing: 0.1em;
        font-family: 'Orbitron', monospace;
    }

    .particle {
        position: fixed;
        border-radius: 50%;
        background: rgba(59,130,246,0.6);
        animation: particleDrift linear infinite;
        pointer-events: none;
        z-index: 1;
    }
    @keyframes particleDrift {
        0% { transform: translateY(100vh) translateX(0) scale(0); opacity: 0; }
        10% { opacity: 1; transform: translateY(80vh) translateX(10px) scale(1); }
        90% { opacity: 0.6; }
        100% { transform: translateY(-20vh) translateX(-30px) scale(0.3); opacity: 0; }
    }

    .scanline {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(59,130,246,0.5), transparent);
        animation: scanDown 6s linear infinite;
        z-index: 2;
        pointer-events: none;
    }
    @keyframes scanDown {
        0% { top: -2px; } 100% { top: 100vh; }
    }

    .version-tag {
        font-size: 0.7rem;
        color: rgba(71,85,105,0.8);
        letter-spacing: 0.15em;
        text-align: center;
        animation: fadeIn 1.5s ease both 1.3s;
        font-family: 'Orbitron', monospace;
    }
    </style>

    <div class="scanline"></div>
    <canvas id="landing-canvas"></canvas>

    <div class="particle" style="left:8%;width:3px;height:3px;animation-duration:8s;animation-delay:0s;"></div>
    <div class="particle" style="left:18%;width:2px;height:2px;animation-duration:11s;animation-delay:2s;"></div>
    <div class="particle" style="left:35%;width:4px;height:4px;animation-duration:9s;animation-delay:1s;background:rgba(167,139,250,0.5);"></div>
    <div class="particle" style="left:55%;width:2px;height:2px;animation-duration:13s;animation-delay:3s;"></div>
    <div class="particle" style="left:72%;width:3px;height:3px;animation-duration:10s;animation-delay:0.5s;background:rgba(52,211,153,0.5);"></div>
    <div class="particle" style="left:88%;width:2px;height:2px;animation-duration:7s;animation-delay:4s;"></div>
    <div class="particle" style="left:45%;width:5px;height:5px;animation-duration:14s;animation-delay:1.5s;background:rgba(59,130,246,0.3);"></div>

    <div class="landing-wrap">

        <div class="logo-ring">
            <span class="logo-emoji">🧬</span>
        </div>

        <h1 class="landing-title">AI-MRI HUB</h1>
        <p class="landing-subtitle">Multidimensional Resistance Intelligence Platform</p>
        <div class="landing-divider"></div>

        <div class="about-grid">
            <div class="about-card">
                <span class="about-card-icon">🎯</span>
                <div class="about-card-title">What is AI-MRI Hub?</div>
                <p class="about-card-text">
                    A state-of-the-art genomic analysis platform that quantifies antibiotic resistance using the
                    <strong style="color:#60a5fa;">Multidimensional Resistance Index (MRI)</strong> —
                    transforming raw gene data into actionable clinical risk scores.
                </p>
            </div>
            <div class="about-card">
                <span class="about-card-icon">🤖</span>
                <div class="about-card-title">AI-Powered Engine</div>
                <p class="about-card-text">
                    Powered by a <strong style="color:#a78bfa;">Random Forest classifier</strong> trained on
                    resistance gene populations, paired with <strong style="color:#34d399;">Gemini AI (J.A.R.V.I.S.)</strong>
                    for natural-language genomic reasoning and clinical Q&amp;A.
                </p>
            </div>
            <div class="about-card">
                <span class="about-card-icon">📊</span>
                <div class="about-card-title">Why It Matters</div>
                <p class="about-card-text">
                    Traditional tools simply list genes. AI-MRI Hub <strong style="color:#60a5fa;">computes,
                    visualizes, and explains</strong> resistance density, mechanism diversity, and clinical
                    susceptibility zones — enabling smarter treatment prioritization.
                </p>
            </div>
        </div>

        <div class="pipeline-wrap">
            <p class="section-label">⬡ How It Works</p>
            <div class="pipeline">
                <div class="pipe-step">
                    <div class="pipe-num">01</div>
                    <span class="pipe-icon">📂</span>
                    <div class="pipe-label">Upload JSON Genome File</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-num">02</div>
                    <span class="pipe-icon">🔬</span>
                    <div class="pipe-label">Extract ARGs &amp; Resistance Genes</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-num">03</div>
                    <span class="pipe-icon">🧮</span>
                    <div class="pipe-label">Compute MRI &amp; ARI Scores</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-num">04</div>
                    <span class="pipe-icon">🤖</span>
                    <div class="pipe-label">AI Risk Classification</div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-num">05</div>
                    <span class="pipe-icon">📋</span>
                    <div class="pipe-label">Generate PDF Clinical Report</div>
                </div>
            </div>
        </div>

        <div class="team-wrap">
            <p class="section-label">⬡ Research &amp; Development Team</p>
            <div class="team-grid">
                <div class="team-pill"><div class="team-avatar">HA</div><span class="team-name">Hardik Agrawal</span></div>
                <div class="team-pill"><div class="team-avatar">PD</div><span class="team-name">Poorva Dongarkar</span></div>
                <div class="team-pill"><div class="team-avatar">YP</div><span class="team-name">Yashraj Patil</span></div>
                <div class="team-pill"><div class="team-avatar">AL</div><span class="team-name">Avani Laswante</span></div>
                <div class="team-pill"><div class="team-avatar">ZB</div><span class="team-name">Zeel Bhanushali</span></div>
                <div class="team-pill"><div class="team-avatar">AW</div><span class="team-name">Aayushi Wasnik</span></div>
                <div class="team-pill"><div class="team-avatar">IP</div><span class="team-name">Indranil Patil</span></div>
            </div>
        </div>

        <div class="enter-btn-wrap">
            <p class="enter-btn-sub">🔬 READY TO ANALYZE</p>
        </div>

        <p class="version-tag">AI-MRI HUB &nbsp;|&nbsp; v2.0 Quantum Edition &nbsp;|&nbsp; Bioinformatics Intelligence Suite</p>
    </div>

    <script>
    const canvas = document.getElementById('landing-canvas');
    const ctx = canvas.getContext('2d');
    let W, H, t = 0;

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    function drawHelix() {
        ctx.clearRect(0, 0, W, H);
        const cx = W / 2;
        const amplitude = Math.min(W * 0.12, 120);
        const freq = 0.018;
        const speed = 0.02;

        for (let strand = 0; strand < 2; strand++) {
            const phaseOffset = strand * Math.PI;
            ctx.beginPath();
            for (let y = -28; y < H + 28; y += 2) {
                const x = cx + amplitude * Math.sin(freq * y + t + phaseOffset);
                if (y === -28) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            const grad = ctx.createLinearGradient(0, 0, 0, H);
            grad.addColorStop(0, 'rgba(59,130,246,0)');
            grad.addColorStop(0.3, strand === 0 ? 'rgba(59,130,246,0.35)' : 'rgba(167,139,250,0.25)');
            grad.addColorStop(0.7, strand === 0 ? 'rgba(59,130,246,0.35)' : 'rgba(52,211,153,0.2)');
            grad.addColorStop(1, 'rgba(59,130,246,0)');
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        for (let y = 40; y < H; y += 40) {
            const x1 = cx + amplitude * Math.sin(freq * y + t);
            const x2 = cx + amplitude * Math.sin(freq * y + t + Math.PI);
            const alpha = 0.08 + 0.06 * Math.sin(y * 0.03 + t * 2);
            ctx.beginPath();
            ctx.moveTo(x1, y);
            ctx.lineTo(x2, y);
            ctx.strokeStyle = `rgba(59,130,246,${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(x1, y, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(96,165,250,${alpha * 2.5})`;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x2, y, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(167,139,250,${alpha * 2.5})`;
            ctx.fill();
        }

        for (let gx = 60; gx < W; gx += 90) {
            for (let gy = 60; gy < H; gy += 90) {
                ctx.beginPath();
                ctx.arc(gx, gy, 0.8, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(59,130,246,0.12)';
                ctx.fill();
            }
        }

        t += speed;
        requestAnimationFrame(drawHelix);
    }
    drawHelix();
    </script>
    """, unsafe_allow_html=True)

    col_c = st.columns([1, 2, 1])[1]
    with col_c:
        if st.button("⬡  ENTER THE PLATFORM  ⬡", type="primary", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()

    st.stop()

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
    if mri < 0.15: return "LOW", " 🟢 "
    elif mri < 0.35: return "MODERATE", " 🟡 "
    else: return "HIGH", " 🔴 "

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
Traditional genomic analysis often simply lists detected genes, which fails to quantify the actual danger a pathogen poses.
Our framework utilizes two calculated metrics to solve this:<br/><br/>
<b>Why ARI is Required:</b> The Antibiotic Resistance Index (ARI) calculates the <i>density</i> of the threat by normalizing the unique mechanisms against the total gene count.
It reveals how efficiently the bacteria utilizes its genetic payload to resist treatments.<br/><br/>
<b>Why MRI Helps:</b> The Multidimensional Resistance Index (MRI) mathematically consolidates the diversity of resisted drugs and deployed mechanisms into a single, standardized risk score.
This allows clinicians and researchers to instantly gauge and compare the severity of different strains, prioritizing high-risk pathogens for immediate intervention without needing to manually decipher complex gene ledgers.
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
st.markdown("""
<div class="main-header">
<h1 style='margin:0; font-size: 2.8rem;'> 🧬  AI-Driven Multidimensional Resistance Index</h1>
<p style='font-size: 1.3rem; opacity: 0.9; margin-top: 10px;'>Quantitative Bio-Analysis of Antibiotic Resistance Genes</p>
<hr style='border: 0.5px solid rgba(255,255,255,0.2); margin: 20px auto; width: 80%;'>
<p style='font-size: 0.95rem; font-weight: 300;'>
<b>Developed by:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil
</p>
</div>
""", unsafe_allow_html=True)

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.current_session = "Chat 1"
    st.session_state.chat_counter = 1

with st.sidebar:
    st.header(" 🗄️ Database Sync")
    if st.button(" 🔄 Refresh Database"):
        st.rerun()

    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])

    if analysis_mode == "Select Known Bacteria":
        selected_file = st.selectbox("Select a Genome:", json_files)

    st.markdown("---")
    st.success(" ✅ AI Brain Connected")

if analysis_mode == "Select Known Bacteria" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    habitat = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)

    if mri > 0.6:
        st.markdown(f'<div class="alert-banner">⚠️ CRITICAL ALERT: {selected_file} identified as High-Priority Superbug (MRI: {round(mri, 3)})</div>', unsafe_allow_html=True)

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
        " ℹ️ Pathogen Summary",
        " 📊 6-Panel Dashboard",
        " 🧮 Math & Data Ledger",
        " 🕸️ Network",
        " 🤖 Bio-AI Chat",
        " 🩺 Clinical Insight",
        " 📄 Export Master PDF",
        " 🌌 3D Landscape"
    ])

    with tab1:
        st.markdown("""
        <div class="welcome-hero">
            <h2 style='text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>Welcome to the World of Bio-Intelligence</h2>
            <p style='color: #e2e8f0;'>Advanced Genomic Analysis Platform for Antimicrobial Resistance (AMR)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###  🧬 System Overview & Intelligence Capabilities")
        st.info("""
        **The AI-MRI Hub provides a state-of-the-art multidimensional analysis suite:**
        * **Genomic ARG Profiling:** Absolute identification and Title-Case formatting of Antibiotic Resistance Genes from sequence data.
        * **Multidimensional Resistance Index (MRI):** A unified clinical risk score consolidating drug evasion and mechanism diversity.
        * **Random Forest Risk Prediction:** Machine learning-driven threat classification based on global population benchmarks.
        * **Safe-Zone Susceptibility Analysis:** Clinical exclusion logic identifying drug classes with zero resistance markers.
        * **Interactive Landscape Mapping:** 3D and 2D spatial visualizations of mechanism diversity and genetic density.
        """)

        st.markdown("###  🦠 Detailed Pathogen Profile")
        st.markdown(f"""
<div class="report-card">
<div class="report-header">Bacterial Identification Ledger</div>
<div class="report-row"><span class="report-label">Target Genome</span><span class="report-value" style="color:#60a5fa;">{selected_file}</span></div>
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

        st.markdown("###  🎯 Metric Explanations & Significance")
        st.info("""
**Pathogen Profile:** Provides the biological and ecological context (Gram, Disease, Habitat) of the strain.
**Total Genes:** The absolute count of Antibiotic Resistance Genes (ARGs) identified.
**Resistance & Mechanisms:** The distinct drug classes evaded and the biological strategies deployed.
**AI Prediction:** A machine learning probability assessment of the overall threat level.

**The Clinical Necessity of MRI and ARI:** Traditional analysis simply lists detected genes.
The **ARI** calculates the *density* and efficiency of the threat relative to the gene count.
The **MRI** mathematically consolidates the diversity of resisted drugs and mechanisms into a single, standardized risk score, allowing researchers to instantly gauge severity and prioritize high-risk pathogens without deciphering complex gene ledgers.
""")

    with tab2:
        st.markdown(f"### Systems Overview: `{selected_file}`")
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)

    with tab3:
        st.markdown("###  🧮 Mathematical Validation")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown('<div class="math-card">', unsafe_allow_html=True)
            st.markdown('<div class="math-card-header">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
            st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
            st.markdown('<div class="annotation-box">'
                '<div class="annotation-item"><span class="annotation-key">U_drugs</span>: Unique Drug Classes Resisted</div>'
                '<div class="annotation-item"><span class="annotation-key">U_mechs</span>: Unique Mechanisms Deployed</div>'
                '<div class="annotation-item"><span class="annotation-key">T_drugs</span>: Total Drug Records Found in sequence</div>'
                '<div class="annotation-item"><span class="annotation-key">T_mechs</span>: Total Mechanism Records Found in sequence</div>'
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

        st.markdown("###  🎯 Risk Assessment Reasoning")
        st.markdown(f"""
        <div class="reasoning-box">
        {get_risk_reason(level, u_drugs, u_mechs)}
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("###  📜 Comprehensive Gene Ledger")
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
            if st.button(" ➕ New Chat", use_container_width=True):
                st.session_state.chat_counter += 1
                new_chat_name = f"Chat {st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_chat_name] = []
                st.session_state.current_session = new_chat_name
                st.rerun()
        with colC:
            st.write("")
            st.write("")
            if st.button(" 🗑️ Clear This Chat", use_container_width=True):
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
                st.error(" ⚠️ AI Library missing. Check your terminal installation.")
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
                        st.error(" ⚠️ **Quota Exceeded (HTTP 429 Error).** The Gemini Free Tier allows a limited number of requests per minute. Please wait 60 seconds and try your question again.")
                    else:
                        st.error(f"AI Connection Error: {e}")

    with tab6:
        st.markdown("### 🩺 Clinical Actionability (Susceptibility Zone)")
        DRUG_UNIVERSE = ["Penicillin", "Cephalosporin", "Carbapenem", "Macrolide", "Aminoglycoside",
                         "Fluoroquinolone", "Tetracycline", "Sulfonamide", "Glycopeptide"]
        resisted_norm = set([d.lower() for d in drug])
        safe_zones = [d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
        st.write("Based on genomic exclusion, the following drug classes show **Zero Resistance Markers** in this sample:")
        st.markdown(f'<div class="susceptibility-card">🛡️ Recommended Target Classes: {", ".join(safe_zones)}</div>', unsafe_allow_html=True)
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

    with tab7:
        st.markdown("###  📥 Generate Complete Master Report")

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

    with tab8:
        st.markdown("###  🌌 Interactive Global Landscape Comparison (Plotly 3D)")
        st.write("You can rotate, zoom, and download this 3D map using the camera icon in the top right corner of the plot.")
        plot_3d_pca_plotly(selected_file)

elif analysis_mode == "AI Predict Unknown":
    st.header(" 🤖 Machine Learning Risk Prediction")
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
