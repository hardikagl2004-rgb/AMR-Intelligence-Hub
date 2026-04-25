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
# LANDING PAGE — uses components.html for
# full CSS/JS/Canvas support (no sanitizing)
# ==========================================
if 'show_landing' not in st.session_state:
    st.session_state.show_landing = True

# Check if user clicked "Enter" via query param
qp = st.query_params
if qp.get("enter") == "1":
    st.session_state.show_landing = False
    st.query_params.clear()

if st.session_state.show_landing:

    # Hide sidebar and all Streamlit chrome
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stHeader"]  { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    footer { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

    LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
    width: 100%; min-height: 100vh;
    background: radial-gradient(ellipse at 50% 20%, #001529 0%, #000b18 50%, #000 100%);
    font-family: 'Rajdhani', sans-serif;
    color: #e2e8f0;
    overflow-x: hidden;
}

canvas#bg {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0; pointer-events: none;
}

.scanline {
    position: fixed; top: -2px; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.6), transparent);
    animation: scanDown 7s linear infinite;
    z-index: 1; pointer-events: none;
}
@keyframes scanDown { 0%{top:-2px} 100%{top:100vh} }

.particle {
    position: fixed; border-radius: 50%;
    pointer-events: none; z-index: 1;
    animation: rise linear infinite;
}
@keyframes rise {
    0%   { transform: translateY(100vh) scale(0); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 0.5; }
    100% { transform: translateY(-10vh) scale(0.4); opacity: 0; }
}

.wrap {
    position: relative; z-index: 10;
    display: flex; flex-direction: column;
    align-items: center; justify-content: flex-start;
    min-height: 100vh; padding: 50px 24px 60px;
}

/* --- LOGO --- */
.logo-outer {
    width: 120px; height: 120px; border-radius: 50%;
    border: 2px solid rgba(59,130,246,0.35);
    display: flex; align-items: center; justify-content: center;
    position: relative; margin-bottom: 30px;
    animation: ringPulse 3s ease-in-out infinite;
    box-shadow: 0 0 50px rgba(59,130,246,0.2);
}
.logo-outer::before {
    content:''; position:absolute; inset:-10px; border-radius:50%;
    border:1px solid rgba(59,130,246,0.12);
    animation: ringPulse 3s ease-in-out infinite 0.6s;
}
.logo-outer::after {
    content:''; position:absolute; inset:-22px; border-radius:50%;
    border:1px solid rgba(59,130,246,0.06);
    animation: ringPulse 3s ease-in-out infinite 1.2s;
}
.logo-icon { font-size: 56px; animation: float 4s ease-in-out infinite; }
@keyframes ringPulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.05);opacity:0.8} }
@keyframes float      { 0%,100%{transform:translateY(0)}       50%{transform:translateY(-7px)} }

/* --- TITLE --- */
.title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.2rem, 6vw, 4rem);
    font-weight: 900;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 45%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center; letter-spacing: 0.04em;
    animation: slideUp 0.9s cubic-bezier(.22,1,.36,1) both;
    margin-bottom: 10px;
}
.subtitle {
    font-size: clamp(0.85rem, 2vw, 1.2rem);
    color: #7dd3fc; letter-spacing: 0.25em;
    text-transform: uppercase; text-align: center;
    animation: slideUp 1.1s cubic-bezier(.22,1,.36,1) both 0.1s;
    margin-bottom: 36px;
}
@keyframes slideUp { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }

.divider {
    width: 180px; height: 1px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    margin: 0 auto 44px;
    animation: fadeIn 1.4s ease both 0.3s;
}
@keyframes fadeIn { from{opacity:0} to{opacity:1} }

/* --- CARDS GRID --- */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 18px; max-width: 960px; width: 100%;
    margin-bottom: 44px;
    animation: fadeIn 1.2s ease both 0.5s;
}
.card {
    background: rgba(10,20,40,0.88);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 16px; padding: 26px 22px;
    backdrop-filter: blur(12px);
    position: relative; overflow: hidden;
    transition: transform .3s ease, border-color .3s ease, box-shadow .3s ease;
}
.card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#3b82f6,#8b5cf6);
    opacity:0; transition:opacity .3s ease;
}
.card:hover { transform:translateY(-7px); border-color:rgba(59,130,246,0.5); box-shadow:0 20px 40px rgba(59,130,246,0.14); }
.card:hover::before { opacity:1; }
.card-icon  { font-size:28px; display:block; margin-bottom:12px; }
.card-title {
    font-family:'Orbitron',monospace; font-size:0.8rem; font-weight:700;
    color:#60a5fa; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;
}
.card-text  { font-size:0.92rem; color:#cbd5e1; line-height:1.72; font-weight:300; }
.hl-blue    { color:#60a5fa; font-weight:600; }
.hl-purple  { color:#a78bfa; font-weight:600; }
.hl-green   { color:#34d399; font-weight:600; }

/* --- PIPELINE --- */
.section-wrap { max-width:960px; width:100%; margin-bottom:44px; animation:fadeIn 1.2s ease both 0.7s; }
.section-label {
    font-family:'Orbitron',monospace; font-size:0.72rem;
    letter-spacing:0.28em; color:#3b82f6;
    text-transform:uppercase; text-align:center; margin-bottom:26px;
}
.pipeline { display:flex; flex-wrap:wrap; justify-content:center; gap:10px; }
.step {
    flex:1; min-width:130px; max-width:175px;
    background:rgba(10,20,40,0.8);
    border:1px solid rgba(59,130,246,0.18);
    border-radius:12px; padding:18px 14px;
    text-align:center; transition:all .3s ease;
}
.step:hover { border-color:#3b82f6; background:rgba(30,58,138,0.28); }
.step-num {
    font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:900;
    color:rgba(59,130,246,0.28); line-height:1; margin-bottom:8px;
    transition:color .3s ease;
}
.step:hover .step-num { color:rgba(59,130,246,0.7); }
.step-icon  { font-size:20px; margin-bottom:8px; display:block; }
.step-label { font-size:0.78rem; font-weight:600; color:#e2e8f0; line-height:1.4; }

/* --- TEAM --- */
.team-grid { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; }
.pill {
    background:rgba(10,20,40,0.9);
    border:1px solid rgba(59,130,246,0.22);
    border-radius:40px; padding:9px 18px;
    display:flex; align-items:center; gap:10px;
    transition:all .3s ease;
}
.pill:hover { background:rgba(30,58,138,0.45); border-color:#60a5fa; transform:scale(1.05); }
.avatar {
    width:32px; height:32px; border-radius:50%;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    display:flex; align-items:center; justify-content:center;
    font-size:0.68rem; font-weight:700; color:#fff; flex-shrink:0;
}
.pname { font-size:0.88rem; color:#e2e8f0; font-weight:600; }

/* --- ENTER BUTTON --- */
.btn-section { margin:10px 0 14px; text-align:center; animation:fadeIn 1.3s ease both 1.1s; }
.ready-label {
    font-family:'Orbitron',monospace; font-size:0.72rem;
    color:#475569; letter-spacing:0.18em; margin-bottom:18px;
}
.enter-btn {
    display:inline-block; padding:18px 56px;
    background:linear-gradient(135deg,#1d4ed8,#2563eb);
    color:#fff; font-family:'Orbitron',monospace;
    font-size:0.88rem; font-weight:700; letter-spacing:0.14em;
    text-transform:uppercase; border-radius:50px;
    border:1px solid rgba(96,165,250,0.4);
    cursor:pointer; position:relative; overflow:hidden;
    transition:all .4s cubic-bezier(.22,1,.36,1);
    box-shadow:0 0 30px rgba(37,99,235,0.45), 0 0 60px rgba(37,99,235,0.18);
    text-decoration:none;
}
.enter-btn::before {
    content:''; position:absolute; top:50%; left:50%;
    width:0; height:0; background:rgba(255,255,255,0.14);
    border-radius:50%; transform:translate(-50%,-50%);
    transition:width .6s ease, height .6s ease;
}
.enter-btn:hover {
    transform:scale(1.08);
    box-shadow:0 0 55px rgba(37,99,235,0.75), 0 0 110px rgba(37,99,235,0.28);
}
.enter-btn:hover::before { width:450px; height:450px; }
.enter-btn:active { transform:scale(0.97); }

.version {
    font-family:'Orbitron',monospace; font-size:0.65rem;
    color:rgba(71,85,105,0.7); letter-spacing:0.15em;
    text-align:center; margin-top:24px;
    animation:fadeIn 1.5s ease both 1.3s;
}
</style>
</head>
<body>

<canvas id="bg"></canvas>
<div class="scanline"></div>

<!-- Floating particles -->
<div class="particle" style="left:7%;width:3px;height:3px;background:rgba(59,130,246,0.7);animation-duration:9s;animation-delay:0s;"></div>
<div class="particle" style="left:19%;width:2px;height:2px;background:rgba(59,130,246,0.5);animation-duration:12s;animation-delay:2s;"></div>
<div class="particle" style="left:34%;width:4px;height:4px;background:rgba(167,139,250,0.6);animation-duration:10s;animation-delay:1s;"></div>
<div class="particle" style="left:54%;width:2px;height:2px;background:rgba(59,130,246,0.5);animation-duration:14s;animation-delay:3.5s;"></div>
<div class="particle" style="left:71%;width:3px;height:3px;background:rgba(52,211,153,0.6);animation-duration:11s;animation-delay:0.7s;"></div>
<div class="particle" style="left:87%;width:2px;height:2px;background:rgba(59,130,246,0.5);animation-duration:8s;animation-delay:4.5s;"></div>
<div class="particle" style="left:46%;width:5px;height:5px;background:rgba(59,130,246,0.3);animation-duration:15s;animation-delay:2s;"></div>

<div class="wrap">

    <!-- LOGO -->
    <div class="logo-outer"><span class="logo-icon">🧬</span></div>

    <!-- TITLE -->
    <h1 class="title">AI-MRI HUB</h1>
    <p class="subtitle">Multidimensional Resistance Intelligence Platform</p>
    <div class="divider"></div>

    <!-- ABOUT CARDS -->
    <div class="grid">
        <div class="card">
            <span class="card-icon">🎯</span>
            <div class="card-title">What is AI-MRI Hub?</div>
            <p class="card-text">
                A state-of-the-art genomic analysis platform that quantifies antibiotic resistance
                using the <span class="hl-blue">Multidimensional Resistance Index (MRI)</span> —
                transforming raw gene data into actionable clinical risk scores.
            </p>
        </div>
        <div class="card">
            <span class="card-icon">🤖</span>
            <div class="card-title">AI-Powered Engine</div>
            <p class="card-text">
                Driven by a <span class="hl-purple">Random Forest classifier</span> trained on
                global resistance gene populations, paired with
                <span class="hl-green">Gemini AI (J.A.R.V.I.S.)</span> for natural-language
                genomic reasoning and clinical Q&A.
            </p>
        </div>
        <div class="card">
            <span class="card-icon">📊</span>
            <div class="card-title">Why It Matters</div>
            <p class="card-text">
                Traditional tools simply list genes. AI-MRI Hub
                <span class="hl-blue">computes, visualizes, and explains</span>
                resistance density, mechanism diversity, and clinical susceptibility zones —
                enabling smarter treatment prioritization.
            </p>
        </div>
    </div>

    <!-- PIPELINE -->
    <div class="section-wrap">
        <p class="section-label">⬡ How It Works</p>
        <div class="pipeline">
            <div class="step"><div class="step-num">01</div><span class="step-icon">📂</span><div class="step-label">Upload JSON Genome File</div></div>
            <div class="step"><div class="step-num">02</div><span class="step-icon">🔬</span><div class="step-label">Extract ARGs &amp; Resistance Genes</div></div>
            <div class="step"><div class="step-num">03</div><span class="step-icon">🧮</span><div class="step-label">Compute MRI &amp; ARI Scores</div></div>
            <div class="step"><div class="step-num">04</div><span class="step-icon">🤖</span><div class="step-label">AI Risk Classification</div></div>
            <div class="step"><div class="step-num">05</div><span class="step-icon">📋</span><div class="step-label">Generate PDF Clinical Report</div></div>
        </div>
    </div>

    <!-- TEAM -->
    <div class="section-wrap" style="animation-delay:0.9s;">
        <p class="section-label">⬡ Research &amp; Development Team</p>
        <div class="team-grid">
            <div class="pill"><div class="avatar">HA</div><span class="pname">Hardik Agrawal</span></div>
            <div class="pill"><div class="avatar">PD</div><span class="pname">Poorva Dongarkar</span></div>
            <div class="pill"><div class="avatar">YP</div><span class="pname">Yashraj Patil</span></div>
            <div class="pill"><div class="avatar">AL</div><span class="pname">Avani Laswante</span></div>
            <div class="pill"><div class="avatar">ZB</div><span class="pname">Zeel Bhanushali</span></div>
            <div class="pill"><div class="avatar">AW</div><span class="pname">Aayushi Wasnik</span></div>
            <div class="pill"><div class="avatar">IP</div><span class="pname">Indranil Patil</span></div>
        </div>
    </div>

    <!-- ENTER BUTTON -->
    <div class="btn-section">
        <p class="ready-label">🔬 READY TO ANALYZE</p>
        <a class="enter-btn" id="enterBtn" href="#">⬡ &nbsp; ENTER THE PLATFORM &nbsp; ⬡</a>
    </div>

    <p class="version">AI-MRI HUB &nbsp;|&nbsp; v2.0 Quantum Edition &nbsp;|&nbsp; Bioinformatics Intelligence Suite</p>

</div>

<script>
// ---- DNA HELIX BACKGROUND ----
const canvas = document.getElementById('bg');
const ctx = canvas.getContext('2d');
let W, H, t = 0;

function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

function draw() {
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2;
    const amp  = Math.min(W * 0.13, 130);
    const freq = 0.017;

    // Two strands
    for (let s = 0; s < 2; s++) {
        const phase = s * Math.PI;
        ctx.beginPath();
        for (let y = -30; y < H + 30; y += 2) {
            const x = cx + amp * Math.sin(freq * y + t + phase);
            y === -30 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        const g = ctx.createLinearGradient(0, 0, 0, H);
        g.addColorStop(0,   'rgba(59,130,246,0)');
        g.addColorStop(0.3, s===0 ? 'rgba(59,130,246,0.38)' : 'rgba(167,139,250,0.26)');
        g.addColorStop(0.7, s===0 ? 'rgba(59,130,246,0.38)' : 'rgba(52,211,153,0.22)');
        g.addColorStop(1,   'rgba(59,130,246,0)');
        ctx.strokeStyle = g;
        ctx.lineWidth   = 1.5;
        ctx.stroke();
    }

    // Base pairs + nodes
    for (let y = 40; y < H; y += 40) {
        const x1 = cx + amp * Math.sin(freq * y + t);
        const x2 = cx + amp * Math.sin(freq * y + t + Math.PI);
        const a  = 0.07 + 0.06 * Math.sin(y * 0.03 + t * 2);
        ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y);
        ctx.strokeStyle = `rgba(59,130,246,${a})`; ctx.lineWidth = 1; ctx.stroke();
        [x1, x2].forEach((x, i) => {
            ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI*2);
            ctx.fillStyle = i===0 ? `rgba(96,165,250,${a*2.8})` : `rgba(167,139,250,${a*2.8})`;
            ctx.fill();
        });
    }

    // Background grid dots
    for (let gx = 60; gx < W; gx += 90)
        for (let gy = 60; gy < H; gy += 90) {
            ctx.beginPath(); ctx.arc(gx, gy, 0.8, 0, Math.PI*2);
            ctx.fillStyle = 'rgba(59,130,246,0.11)'; ctx.fill();
        }

    t += 0.018;
    requestAnimationFrame(draw);
}
draw();

// ---- ENTER BUTTON → tell parent Streamlit to navigate ----
document.getElementById('enterBtn').addEventListener('click', function(e) {
    e.preventDefault();
    // Send message to the parent Streamlit window
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'enter'}, '*');
});
</script>
</body>
</html>"""

    # Render the full landing page in a true iframe
    components.html(LANDING_HTML, height=900, scrolling=True)

    # Listen for the postMessage from inside the iframe
    # Use a tiny JS snippet in the parent to catch it and trigger st rerun via query param
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'streamlit:setComponentValue' && e.data.value === 'enter') {
            // Navigate to ?enter=1 which Streamlit picks up on rerun
            window.location.search = '?enter=1';
        }
    });
    </script>
    """, unsafe_allow_html=True)

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
            gene_name = inner.get("ARO_name", k).upper()
            d, m = [], []
            for c in inner.get("ARO_category", {}).values():
                cname = c.get("category_aro_class_name","").lower()
                val   = c.get("category_aro_name","").title()
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            drug.extend(d); mech.extend(m)
            records.append((gene_name, ", ".join(set(d)), ", ".join(set(m)), habitat.title()))
        except:
            continue
    genes = len(data)
    mri = (len(set(drug)) + len(set(mech))) / (len(drug) + len(mech) + 1)
    ari = len(set(mech)) / (genes + 1)
    return genes, drug, mech, mri, ari, records

def get_level(mri):
    if mri < 0.15:   return "LOW",      "🟢"
    elif mri < 0.35: return "MODERATE", "🟡"
    else:            return "HIGH",      "🔴"

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
                X.append(features); y.append(label)
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
        a.set_facecolor('#0e1117'); a.tick_params(colors='white'); a.title.set_color('white')
    drug_c = Counter(drug).most_common(8)
    if drug_c:
        ax[0,0].pie([v for k,v in drug_c], labels=[k[:15]+".." for k,v in drug_c], autopct='%1.1f%%', textprops={'color':"w"})
    ax[0,0].set_title("Drug Classes Resisted")
    mech_c = Counter(mech)
    if mech_c:
        ax[0,1].bar([k[:15]+".." for k in mech_c.keys()], mech_c.values(), color=color)
        ax[0,1].tick_params(axis='x', rotation=35)
    ax[0,1].set_title("Mechanisms Deployed")
    theta = np.linspace(0, np.pi, 100)
    ax[0,2].plot(np.cos(theta), np.sin(theta), color='gray')
    ang = mri * np.pi
    ax[0,2].plot([0, np.cos(ang)], [0, np.sin(ang)], color=color, linewidth=5)
    ax[0,2].axis('off'); ax[0,2].set_title(f"MRI Indicator: {round(mri,3)} ({level})")
    gene_list = [r[0] for r in records]; gene_c = Counter(gene_list).most_common(5)
    if gene_c:
        ax[1,0].bar([k[:15]+".." for k,v in gene_c], [v for k,v in gene_c], color='#87CEEB')
        ax[1,0].tick_params(axis='x', rotation=35)
    ax[1,0].set_title("Top Gene Frequency")
    ax[1,1].bar(["Total Genes"], [genes], color='#2E86C1'); ax[1,1].set_title("Overall Gene Count")
    ax[1,2].bar(["Unique Drugs","Unique Mechs"], [len(set(drug)),len(set(mech))], color=["#9B59B6","#E67E22"])
    ax[1,2].set_title("Diversity Comparison")
    fig.tight_layout()
    return fig

def generate_network_html(records, organism_name, color):
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white',
                  cdn_resources="in_line", select_menu=True, filter_menu=True)
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
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(net.generate_html())
    return html_path

def plot_3d_pca_plotly(current_file):
    X, files, risk_levels = [], [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g, d, m, mr, ar, _ = extract_data(f)
                X.append([mr, len(set(m)), len(set(d))]); files.append(f)
                lv, _ = get_level(mr)
                risk_levels.append("TARGET 🎯" if f == current_file else lv)
            except: continue
    if len(X) < 3:
        st.warning("Not enough data for 3D PCA."); return
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca, columns=['Overall Resistance (PC1)','Mechanism Diversity (PC2)','Genetic Density (PC3)'])
    df_pca['Genome'] = files; df_pca['Risk Category'] = risk_levels
    color_map = {"HIGH":"red","MODERATE":"orange","LOW":"green","TARGET 🎯":"gold"}
    fig = px.scatter_3d(df_pca, x='Overall Resistance (PC1)', y='Mechanism Diversity (PC2)',
                        z='Genetic Density (PC3)', color='Risk Category', hover_name='Genome',
                        color_discrete_map=color_map, opacity=0.8, size_max=10)
    fig.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')), selector=dict(name="TARGET 🎯"))
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='#0e1117', font_color='white',
                      scene=dict(xaxis=dict(backgroundcolor="#0e1117",gridcolor="gray"),
                                 yaxis=dict(backgroundcolor="#0e1117",gridcolor="gray"),
                                 zaxis=dict(backgroundcolor="#0e1117",gridcolor="gray")))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. MASTER PDF GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon, records, dashboard_fig, bac_info, habitat, ai_pred_text, ai_conf_text):
    pdf_file = f"{bac_name.replace('.json','')}_Detailed_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], fontSize=18, spaceAfter=15, textColor=colors.HexColor('#1E3A8A'))
    h2_style     = ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2E86C1'))
    normal_style = styles['Normal']
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    t_drugs, t_mechs = len(drug), len(mech)
    elements = []
    elements.append(Paragraph(f"AI-MRI Report: {bac_name.replace('.json','')}", title_style))
    elements.append(Paragraph("Executive Summary & AI Analysis", h2_style))
    elements.append(Paragraph(f"<b>Pathogen File:</b> {bac_name}<br/><b>Gram Stain:</b> {bac_info['gram']}<br/><b>Common Disease:</b> {bac_info['disease']}<br/><b>Habitat:</b> {habitat}<br/><b>Total Genes:</b> {genes}<br/><b>Resistance (Drugs):</b> {u_drugs}<br/><b>Mechanisms:</b> {u_mechs}<br/><b>MRI Score:</b> {round(mri,3)} ({level})<br/><b>ARI Score:</b> {round(ari,3)}<br/><br/><b>ML AI Prediction:</b> {ai_pred_text}<br/><b>Confidence Matrix:</b> {ai_conf_text}", normal_style))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Metric Explanations & Significance", h2_style))
    elements.append(Paragraph("<b>Pathogen Profile:</b> Provides the biological and ecological context of the strain.<br/><br/><b>Total Genes:</b> The absolute count of ARGs identified.<br/><br/><b>Resistance & Mechanisms:</b> The distinct drug classes evaded and biological strategies deployed.<br/><br/><b>AI Prediction:</b> A Random Forest model's probabilistic threat level assessment.", normal_style))
    elements.append(Paragraph("The Clinical Necessity of MRI and ARI", h2_style))
    elements.append(Paragraph("Traditional genomic analysis lists genes but fails to quantify danger. <b>ARI</b> calculates the <i>density</i> of the threat. <b>MRI</b> consolidates the diversity of resisted drugs and mechanisms into a single standardized score for rapid clinical comparison.", normal_style))
    elements.append(PageBreak())
    elements.append(Paragraph("Exact Mathematical Calculations", h2_style))
    elements.append(Paragraph(f"<b>MRI:</b> ({u_drugs} + {u_mechs}) / ({t_drugs} + {t_mechs} + 1) = <b>{round(mri,3)}</b><br/><b>ARI:</b> {u_mechs} / ({genes} + 1) = <b>{round(ari,3)}</b>", normal_style))
    elements.append(Paragraph("Risk Assessment Reasoning", h2_style))
    elements.append(Paragraph(get_risk_reason(level, u_drugs, u_mechs), normal_style))
    elements.append(Paragraph("Graphical Systems Dashboard", h2_style))
    buf = io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for ax in dashboard_fig.axes:
        ax.set_facecolor('white'); ax.tick_params(colors='black'); ax.title.set_color('black')
        for text in ax.texts: text.set_color('black')
    dashboard_fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    elements.append(RLImage(buf, width=7.5*inch, height=5*inch))
    elements.append(PageBreak())
    elements.append(Paragraph("Complete Gene Ledger", h2_style))
    table_data = [["Gene Name","Drugs Resisted","Mechanisms Used","Habitat"]]
    for g, d, m, h in records:
        table_data.append([Paragraph(g,normal_style),Paragraph(d,normal_style),Paragraph(m,normal_style),Paragraph(h,normal_style)])
    t = Table(table_data, colWidths=[1.2*inch,2.2*inch,2.2*inch,0.9*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2E86C1')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F8F9F9')),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
    ]))
    elements.append(t)
    doc.build(elements)
    return pdf_file

# ==========================================
# 5. FRONTEND: MAIN APP LAYOUT
# ==========================================
st.markdown("""
<div class="main-header">
<h1 style='margin:0; font-size: 2.8rem;'>🧬 AI-Driven Multidimensional Resistance Index</h1>
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
    habitat  = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)

    if mri > 0.6:
        st.markdown(f'<div class="alert-banner">⚠️ CRITICAL ALERT: {selected_file} identified as High-Priority Superbug (MRI: {round(mri,3)})</div>', unsafe_allow_html=True)

    model = train_rf_model()
    ai_pred_text = "N/A"; ai_conf_text = "N/A"
    if model:
        pred = model.predict([[genes, u_drugs, u_mechs]])[0]
        probs = model.predict_proba([[genes, u_drugs, u_mechs]])[0]
        classes = model.classes_
        conf_dict = {str(c): round(float(p),3) for c,p in zip(classes,probs)}
        ai_pred_text = str(pred); ai_conf_text = str(conf_dict).replace("'","")

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{level} {icon}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">MRI Score</div><div class="metric-value">{round(mri,3)}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Genes</div><div class="metric-value">{genes}</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div class="metric-label">Habitat</div><div class="metric-value">{habitat}</div></div>', unsafe_allow_html=True)

    st.write(" ")
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
        "ℹ️ Pathogen Summary","📊 6-Panel Dashboard","🧮 Math & Data Ledger",
        "🕸️ Network","🤖 Bio-AI Chat","🩺 Clinical Insight","📄 Export Master PDF","🌌 3D Landscape"
    ])

    with tab1:
        st.markdown("""
        <div class="welcome-hero">
            <h2>Welcome to the World of Bio-Intelligence</h2>
            <p>Advanced Genomic Analysis Platform for Antimicrobial Resistance (AMR)</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 🧬 System Overview & Intelligence Capabilities")
        st.info("""**The AI-MRI Hub provides a state-of-the-art multidimensional analysis suite:**
* **Genomic ARG Profiling:** Absolute identification and Title-Case formatting of ARGs from sequence data.
* **Multidimensional Resistance Index (MRI):** A unified clinical risk score consolidating drug evasion and mechanism diversity.
* **Random Forest Risk Prediction:** ML-driven threat classification based on global population benchmarks.
* **Safe-Zone Susceptibility Analysis:** Clinical exclusion logic identifying drug classes with zero resistance markers.
* **Interactive Landscape Mapping:** 3D and 2D spatial visualizations of mechanism diversity and genetic density.""")
        st.markdown("### 🦠 Detailed Pathogen Profile")
        st.markdown(f"""<div class="report-card">
<div class="report-header">Bacterial Identification Ledger</div>
<div class="report-row"><span class="report-label">Target Genome</span><span class="report-value" style="color:#60a5fa;">{selected_file}</span></div>
<div class="report-row"><span class="report-label">Gram Classification</span><span class="report-value">{bac_info['gram']}</span></div>
<div class="report-row"><span class="report-label">Associated Pathology</span><span class="report-value">{bac_info['disease']}</span></div>
<div class="report-row"><span class="report-label">Ecological Habitat</span><span class="report-value">{habitat}</span></div>
<div class="report-row"><span class="report-label">Genomic ARG Count</span><span class="report-value">{genes} Genes</span></div>
<div class="report-row"><span class="report-label">Resistance Breadth</span><span class="report-value">{u_drugs} Drug Classes</span></div>
<div class="report-row"><span class="report-label">Deployed Mechanisms</span><span class="report-value">{u_mechs} Strategies</span></div>
<div class="report-row"><span class="report-label">Calculated MRI / ARI</span><span class="report-value">{round(mri,3)} ({level}) / {round(ari,3)}</span></div>
<div style="margin-top:20px;padding-top:10px;">
<span class="report-label">AI Predictive Verdict:</span>
<span class="ai-badge">{ai_pred_text} Risk</span>
<br><br><small style="color:#64748b;">Confidence Matrix: {ai_conf_text}</small>
</div></div>""", unsafe_allow_html=True)
        st.markdown("### 🎯 Metric Explanations & Significance")
        st.info("""**Pathogen Profile:** Provides the biological and ecological context (Gram, Disease, Habitat) of the strain.
**Total Genes:** The absolute count of Antibiotic Resistance Genes (ARGs) identified.
**Resistance & Mechanisms:** The distinct drug classes evaded and the biological strategies deployed.
**AI Prediction:** A machine learning probability assessment of the overall threat level.

The **ARI** calculates the *density* and efficiency of the threat relative to the gene count.
The **MRI** mathematically consolidates the diversity of resisted drugs and mechanisms into a single standardized risk score.""")

    with tab2:
        st.markdown(f"### Systems Overview: `{selected_file}`")
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)

    with tab3:
        st.markdown("### 🧮 Mathematical Validation")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown('<div class="math-card"><div class="math-card-header">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
            st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
            st.markdown('<div class="annotation-box"><div class="annotation-item"><span class="annotation-key">U_drugs</span>: Unique Drug Classes Resisted</div><div class="annotation-item"><span class="annotation-key">U_mechs</span>: Unique Mechanisms Deployed</div><div class="annotation-item"><span class="annotation-key">T_drugs</span>: Total Drug Records in sequence</div><div class="annotation-item"><span class="annotation-key">T_mechs</span>: Total Mechanism Records in sequence</div><div class="annotation-item"><span class="annotation-key">+ 1</span>: Laplace smoothing constant</div></div>', unsafe_allow_html=True)
            st.markdown("**Current Calculation:**")
            st.latex(rf"\frac{{{u_drugs} + {u_mechs}}}{{{len(drug)} + {len(mech)} + 1}} = {round(mri,3)}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown('<div class="math-card"><div class="math-card-header">Antibiotic Resistance Index (ARI)</div>', unsafe_allow_html=True)
            st.latex(r"ARI = \frac{U_{mechs}}{G_{total} + 1}")
            st.markdown('<div class="annotation-box"><div class="annotation-item"><span class="annotation-key">U_mechs</span>: Unique Mechanisms Deployed</div><div class="annotation-item"><span class="annotation-key">G_total</span>: Total Genomic Gene count</div><div class="annotation-item"><span class="annotation-key">+ 1</span>: Laplace smoothing constant</div></div>', unsafe_allow_html=True)
            st.markdown("**Current Calculation:**")
            st.latex(rf"\frac{{{u_mechs}}}{{{genes} + 1}} = {round(ari,3)}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Assessment Reasoning")
        st.markdown(f'<div class="reasoning-box">{get_risk_reason(level, u_drugs, u_mechs)}</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("### 📜 Comprehensive Gene Ledger")
        df = pd.DataFrame(records, columns=["Gene Name","Drug Class","Mechanism","Habitat"])
        st.dataframe(df, use_container_width=True)

    with tab4:
        st.markdown("### Interactive Mechanism Network")
        st.write("Use the filter menu within the interactive map to isolate specific nodes.")
        html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
        with open(html_path,'r',encoding='utf-8') as f:
            components.html(f.read(), height=650)

    with tab5:
        colA,colB,colC = st.columns([0.6,0.2,0.2])
        with colA:
            st.session_state.current_session = st.selectbox("Active Chat Session:", list(st.session_state.chat_sessions.keys()))
        with colB:
            st.write(""); st.write("")
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_counter += 1
                new_name = f"Chat {st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_name] = []
                st.session_state.current_session = new_name
                st.rerun()
        with colC:
            st.write(""); st.write("")
            if st.button("🗑️ Clear This Chat", use_container_width=True):
                st.session_state.chat_sessions[st.session_state.current_session] = []
                st.rerun()
        for msg in st.session_state.chat_sessions[st.session_state.current_session]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        user_msg = st.chat_input(f"Ask me about {selected_file}...")
        if user_msg:
            st.chat_message("user").markdown(user_msg)
            st.session_state.chat_sessions[st.session_state.current_session].append({"role":"user","content":user_msg})
            if not AI_AVAILABLE:
                st.error("⚠️ AI Library missing.")
            else:
                try:
                    context = f"""You are J.A.R.V.I.S., an expert Bioinformatics AI.
The user is analyzing: '{selected_file}'.
Data: Total Genes={genes} | Drugs={u_drugs} | Mechanisms={u_mechs} | MRI={round(mri,3)} ({level}) | ARI={round(ari,3)}
Question: {user_msg}"""
                    with st.spinner("Processing genome logic..."):
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        if not available_models:
                            st.error("No text generation models available.")
                        else:
                            target_model = next((m for m in available_models if 'flash' in m), next((m for m in available_models if 'pro' in m), available_models[0]))
                            response = genai.GenerativeModel(target_model).generate_content(context)
                            st.chat_message("assistant").markdown(response.text)
                            st.session_state.chat_sessions[st.session_state.current_session].append({"role":"assistant","content":response.text})
                except Exception as e:
                    err = str(e)
                    if "429" in err or "quota" in err.lower():
                        st.error("⚠️ Quota Exceeded (HTTP 429). Wait 60 seconds and retry.")
                    else:
                        st.error(f"AI Connection Error: {e}")

    with tab6:
        st.markdown("### 🩺 Clinical Actionability (Susceptibility Zone)")
        DRUG_UNIVERSE = ["Penicillin","Cephalosporin","Carbapenem","Macrolide","Aminoglycoside","Fluoroquinolone","Tetracycline","Sulfonamide","Glycopeptide"]
        resisted_norm = set([d.lower() for d in drug])
        safe_zones = [d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
        st.write("Based on genomic exclusion, the following drug classes show **Zero Resistance Markers**:")
        st.markdown(f'<div class="susceptibility-card">🛡️ Recommended Target Classes: {", ".join(safe_zones)}</div>', unsafe_allow_html=True)
        st.write("---")
        st.markdown("### 📈 Population Benchmark")
        all_mris = []
        for f in json_files:
            try: _, _, _, f_mri, _, _ = extract_data(f); all_mris.append(f_mri)
            except: continue
        if all_mris:
            avg_mri = sum(all_mris)/len(all_mris)
            st.bar_chart(pd.DataFrame({"MRI Score":[mri,avg_mri]}, index=["Target Genome","Global Average"]))

    with tab7:
        st.markdown("### 📥 Generate Complete Master Report")
        if st.button("Generate Master PDF", type="primary"):
            with st.spinner("Compiling report..."):
                pdf_path = create_advanced_pdf_report(selected_file, genes, drug, mech, mri, ari, level, icon, records, fig, bac_info, habitat, ai_pred_text, ai_conf_text)
                with open(pdf_path,"rb") as file:
                    st.download_button(label="Download Detailed PDF Report", data=file, file_name=pdf_path, mime="application/pdf")

    with tab8:
        st.markdown("### 🌌 Interactive Global Landscape Comparison (Plotly 3D)")
        st.write("Rotate, zoom, and download this 3D map using the camera icon.")
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
        prob_str = " ".join([f"{c[0]}:{p:.2f}" for c,p in zip(classes,probs)])
        st.success(f"### AI Prediction: **{prediction}** ({prob_str})")
