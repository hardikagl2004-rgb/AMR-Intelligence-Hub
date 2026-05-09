# =========================
# FINAL CORRECTED COMBINED VERSION
# =========================

import io
import json
import math
import random
import streamlit as st
import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

# -------------------------------------------------------------------
# KEEP THESE FROM CODE 1 EXACTLY AS THEY ALREADY EXIST IN YOUR APP:
# - LABBACTERIADB
# - LABANTIBIOTICS
# - LABDILUTIONS
# - labmicforantibiotic(ab, resistancefactor)
# - labgetrisk(mri)
# - labsimulate(bacterianame, resistanceoverride, temperature, ph, incubationh, cfuexp, medium)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# SMALL EMBEDDED PDF GENERATOR
# This replaces the external "from lab_enhancements import generate_lab_report_pdf"
# so the merged app works as a single file.
# -------------------------------------------------------------------
def generatelabreportpdf(simdata):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "LabTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0f3d5e"),
        spaceAfter=12,
    )
    h_style = ParagraphStyle(
        "LabHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#124e78"),
        spaceAfter=8,
        spaceBefore=8,
    )
    body = styles["BodyText"]
    body.leading = 15

    elements = []
    db = simdata["db"]

    elements.append(Paragraph("Virtual Laboratory Report", title_style))
    elements.append(Paragraph(f"<b>Organism:</b> {simdata['bacteria']}", body))
    elements.append(Paragraph(
        f"<b>Conditions:</b> {simdata['temperature']}°C, pH {simdata['ph']}, "
        f"{simdata['incubation_h']} h, 10^{simdata['cfu_exp']} CFU/mL, {simdata['medium']}",
        body
    ))
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph("1. Overview", h_style))
    overview_data = [
        ["Metric", "Value"],
        ["Lab MRI", str(simdata["lab_mri"])],
        ["AI MRI", str(simdata["ai_mri"])],
        ["Lab ARI", str(simdata["lab_ari"])],
        ["AI ARI", str(simdata["ai_ari"])],
        ["Lab Risk", simdata["lab_risk"][0]],
        ["AI Risk", simdata["ai_risk"][0]],
        ["Estimated ARGs", str(simdata["total_genes"])],
        ["Drug Classes", str(simdata["u_drug_classes"])],
        ["Mechanisms", str(simdata["u_mechs"])],
    ]
    t = Table(overview_data, colWidths=[2.0 * inch, 3.8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)

    elements.append(Paragraph("2. Organism Identity", h_style))
    ident_data = [
        ["Property", "Value"],
        ["Gram stain", db["gram"]],
        ["Morphology", db["shape"]],
        ["Arrangement", db["arrangement"]],
        ["Motility", "Yes" if db["motility"] else "No"],
        ["Spore formation", "Yes" if db["spore"] else "No"],
        ["Capsule", "Present" if db["capsule"] else "Absent"],
        ["Oxidase", db["oxidase"]],
        ["Catalase", db["catalase"]],
        ["Selective medium", db["selective_media"]],
        ["Optimal temperature", f"{db['optimal_temp']}°C"],
        ["Optimal pH", str(db["optimal_ph"])],
        ["Disease association", db["disease"]],
    ]
    t2 = Table(ident_data, colWidths=[2.0 * inch, 3.8 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcfce7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t2)

    elements.append(Paragraph("3. Growth Kinetics", h_style))
    elements.append(Paragraph(
        "Growth values are simulated from the selected organism’s optimal temperature, pH, and resistance pressure. "
        "The growth curve reflects lag, logarithmic, and stationary phases.",
        body
    ))
    growth_data = [["Time", "CFU/mL"]]
    for tpt, cfu in zip(simdata["growth_times"], simdata["growth_cfu"]):
        growth_data.append([str(tpt), str(cfu)])
    tg = Table(growth_data[:16], colWidths=[1.5 * inch, 2.5 * inch])
    tg.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(tg)

    elements.append(Paragraph("4. MIC Results", h_style))
    mic_table = [["Antibiotic", "Class", "Lab MIC", "Lab", "AI MIC", "AI"]]
    for r in simdata["mic_results"]:
        mic_table.append([
            r["name"], r["class"], f"{r['lab_mic']} µg/mL", r["lab_interp"],
            f"{r['ai_mic']} µg/mL", r["ai_interp"]
        ])
    tm = Table(mic_table, repeatRows=1, colWidths=[1.4 * inch, 1.2 * inch, 1.0 * inch, 0.5 * inch, 1.0 * inch, 0.5 * inch])
    tm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0f2fe")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(tm)

    elements.append(PageBreak())
    elements.append(Paragraph("5. Resistance Genes", h_style))
    elements.append(Paragraph(", ".join(db["notable_genes"]), body))

    elements.append(Paragraph("6. AI vs Lab Summary", h_style))
    matched_ab = sum(1 for r in simdata["mic_results"] if r["lab_interp"] == r["ai_interp"])
    total_ab = len(simdata["mic_results"])
    elements.append(Paragraph(
        f"MRI delta: {round(abs(simdata['lab_mri'] - simdata['ai_mri']), 3)}<br/>"
        f"ARI delta: {round(abs(simdata['lab_ari'] - simdata['ai_ari']), 3)}<br/>"
        f"Antibiotic concordance: {matched_ab}/{total_ab} ({round(matched_ab/total_ab*100)}%)",
        body
    ))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# -------------------------------------------------------------------
# ENHANCED HTML BUILDER (REPLACE OLD labbuildhtml)
# -------------------------------------------------------------------
def labbuildhtml(sim):
    import json, math, random

    db = sim["db"]
    gram_col = db["gram_color"]
    colony_c = db["colony_color"]
    risk_lab = sim["lab_risk"]
    risk_ai = sim["ai_risk"]

    growth_labels_js = json.dumps(sim["growth_times"])
    growth_data_js = json.dumps([round(v / 1e6, 2) for v in sim["growth_cfu"]])

    ab_names = [r["name"] for r in sim["mic_results"]]
    lab_mics = [math.log2(max(r["lab_mic"], 0.03)) for r in sim["mic_results"]]
    ai_mics = [math.log2(max(r["ai_mic"], 0.03)) for r in sim["mic_results"]]

    classes = list(dict.fromkeys([r["class"] for r in sim["mic_results"]]))
    lab_profile_vals = []
    ai_profile_vals = []
    for cl in classes:
        lv = [math.log2(max(r["lab_mic"], 0.03)) for r in sim["mic_results"] if r["class"] == cl]
        av = [math.log2(max(r["ai_mic"], 0.03)) for r in sim["mic_results"] if r["class"] == cl]
        lab_profile_vals.append(round(sum(lv) / len(lv), 2) if lv else 0)
        ai_profile_vals.append(round(sum(av) / len(av), 2) if av else 0)

    mri_delta = round(abs(sim["lab_mri"] - sim["ai_mri"]), 3)
    ari_delta = round(abs(sim["lab_ari"] - sim["ai_ari"]), 3)
    mri_match = "IDENTICAL" if mri_delta < 0.03 else "NEAR MATCH" if mri_delta < 0.08 else "DIVERGED"
    risk_agree = sim["lab_risk"][0] == sim["ai_risk"][0]

    matched_ab = sum(1 for r in sim["mic_results"] if r["lab_interp"] == r["ai_interp"])
    total_ab = len(sim["mic_results"])
    match_pct = round(matched_ab / total_ab * 100)

    mic_rows = ""
    for r in sim["mic_results"]:
        lab_cls = {"S": "#3B6D11", "I": "#BA7517", "R": "#A32D2D"}[r["lab_interp"]]
        ai_cls = {"S": "#3B6D11", "I": "#BA7517", "R": "#A32D2D"}[r["ai_interp"]]
        match = r["lab_interp"] == r["ai_interp"]
        match_sym = "check" if match else "near" if abs(r["lab_mic"] - r["ai_mic"]) < r["S_break"] * 2 else "diff"
        match_col = "#3B6D11" if match else "#BA7517" if match_sym == "near" else "#A32D2D"
        match_disp = "&#10003;" if match else "&#8776;" if match_sym == "near" else "&#10007;"
        mic_rows += f"""
        <tr>
          <td style="padding:7px 8px;font-size:12px;word-break:break-word;">{r['name']}</td>
          <td style="padding:7px 8px;font-size:11px;color:#888;word-break:break-word;">{r['class']}</td>
          <td style="padding:7px 8px;font-size:12px;font-weight:500;">{r['lab_mic']} µg/mL</td>
          <td style="padding:7px 8px;"><span style="display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600;background:{lab_cls}22;color:{lab_cls};">{r['lab_interp']}</span></td>
          <td style="padding:7px 8px;font-size:12px;font-weight:500;">{r['ai_mic']} µg/mL</td>
          <td style="padding:7px 8px;"><span style="display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600;background:{ai_cls}22;color:{ai_cls};">{r['ai_interp']}</span></td>
          <td style="padding:7px 8px;font-size:17px;color:{match_col};text-align:center;">{match_disp}</td>
        </tr>
        """

    rng2 = random.Random(hash(sim["bacteria"]) % 9999)
    colony_svg = ""
    for _ in range(sim["colony_count"]):
        cx = rng2.randint(15, 185)
        cy = rng2.randint(15, 185)
        r2 = rng2.randint(3, 9)
        colony_svg += f'<circle cx="{cx}" cy="{cy}" r="{r2}" fill="{colony_c}" stroke="#aaa" stroke-width="0.5" opacity="0.85"/>'

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

    def risk_badge(rt):
        label, col, _ = rt
        return f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;background:{col}22;color:{col};font-size:12px;font-weight:700;">{label}</span>'

    res_bars = ""
    for cl in ["Beta-lactam", "Fluoroquinolone", "Aminoglycoside", "Carbapenem",
               "Tetracycline", "Macrolide", "Glycopeptide", "Colistin"]:
        v = db["base_resistance"].get(cl, 0)
        pct = round(v * 100)
        bar_col = '#ef4444' if v > 0.5 else '#f59e0b' if v > 0.25 else '#22c55e'
        res_bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <span style="font-size:11px;min-width:115px;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{cl}</span>
          <div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{bar_col};border-radius:3px;"></div>
          </div>
          <span style="font-size:11px;min-width:36px;text-align:right;color:#64748b;">{pct}%</span>
        </div>
        """

    gene_tags = "".join(
        f'<span style="display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;font-family:monospace;'
        f'background:rgba(0,201,255,0.08);border:1px solid rgba(0,201,255,0.2);color:#67e8f9;margin:2px;word-break:break-all;">{g}</span>'
        for g in db["notable_genes"]
    )

    alert_html = (
        '<div style="padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px;'
        'background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#f87171;word-wrap:break-word;">'
        '&#9888; HIGH PRIORITY PATHOGEN &mdash; MRI Score exceeds 0.35 threshold</div>'
        if sim["lab_risk"][0] == "HIGH" else
        '<div style="padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px;'
        'background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);color:#fbbf24;word-wrap:break-word;">'
        '&#9889; MODERATE RISK &mdash; Antibiogram-guided therapy recommended</div>'
        if sim["lab_risk"][0] == "MODERATE" else
        '<div style="padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px;'
        'background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:#4ade80;word-wrap:break-word;">'
        '&#10003; LOW RISK &mdash; Standard empirical therapy protocols viable</div>'
    )

    risk_lab_label = risk_lab[0]
    risk_ai_label = risk_ai[0]
    risk_lab_col = risk_lab[1]
    risk_ai_col = risk_ai[1]

    ai_lab_rows = ""
    ai_lab_items = [
        ("MRI Score", f"{sim['lab_mri']} ({risk_lab_label})", f"{sim['ai_mri']} ({risk_ai_label})", mri_delta < 0.03, mri_delta < 0.08),
        ("ARI Score", str(sim["lab_ari"]), str(sim["ai_ari"]), ari_delta < 0.02, ari_delta < 0.06),
        ("Risk Classification", risk_lab_label, risk_ai_label, risk_agree, risk_agree),
        ("MIC Concordance", f"{matched_ab}/{total_ab} ({match_pct}%)", "AI predicted", match_pct >= 80, match_pct >= 60),
    ]
    for r in sim["mic_results"]:
        match = r["lab_interp"] == r["ai_interp"]
        near = abs(r["lab_mic"] - r["ai_mic"]) < r["S_break"] * 2
        ai_lab_items.append((
            r["name"],
            f"{r['lab_mic']} µg/mL ({r['lab_interp']})",
            f"{r['ai_mic']} µg/mL ({r['ai_interp']})",
            match, near
        ))

    for label, lab_v, ai_v, exact, near in ai_lab_items:
        icon = "&#10003;" if exact else "&#8776;" if near else "&#10007;"
        icon_col = "#22c55e" if exact else "#f59e0b" if near else "#ef4444"
        ai_lab_rows += f"""
        <tr>
          <td style="padding:7px 10px;font-size:12px;word-break:break-word;">{label}</td>
          <td style="padding:7px 10px;font-size:12px;color:#60a5fa;">{lab_v}</td>
          <td style="padding:7px 10px;font-size:12px;color:#a78bfa;">{ai_v}</td>
          <td style="padding:7px 10px;font-size:16px;color:{icon_col};text-align:center;">{icon}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Virtual Lab -- {sim['bacteria']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Sora',sans-serif;background:#0a0f1a;color:#e2e8f0;min-height:100vh;word-wrap:break-word;overflow-wrap:break-word;}}
:root{{--accent:#00c9ff;--accent2:#7c3aed;--card:#111827;--border:rgba(255,255,255,0.07);--muted:#64748b}}
.lab-shell{{max-width:1150px;margin:0 auto;padding:20px 16px}}
.lab-header{{background:linear-gradient(135deg,#0c1f3f,#111827,#0f1a2e);border:1px solid var(--border);border-radius:16px;padding:20px 24px;margin-bottom:18px;position:relative;overflow:hidden}}
.lab-header::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00c9ff,#7c3aed,#10b981,#00c9ff);background-size:300% auto;animation:scanBar 4s linear infinite}}
@keyframes scanBar{{0%{{background-position:0% center}}100%{{background-position:300% center}}}}
.bac-name{{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:600;color:#00c9ff;margin-bottom:4px;word-break:break-word;}}
.bac-disease{{font-size:13px;color:var(--muted);margin-top:4px;word-wrap:break-word;line-height:1.5;}}
.tab-nav{{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:18px;overflow-x:auto;flex-wrap:nowrap;}}
.tab-btn{{padding:9px 14px;font-size:12px;font-weight:600;color:var(--muted);background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap;transition:all 0.2s;font-family:'Sora',sans-serif;}}
.tab-btn:hover{{color:#e2e8f0}} .tab-btn.active{{color:#00c9ff;border-bottom-color:#00c9ff}}
.tab-panel{{display:none}} .tab-panel.active{{display:block}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin-bottom:14px}}
.card-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#00c9ff;margin-bottom:12px}}
.explain-box{{background:rgba(0,201,255,0.05);border:1px solid rgba(0,201,255,0.15);border-left:3px solid #00c9ff;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:12px;color:#94a3b8;line-height:1.75;word-wrap:break-word;}}
.explain-box strong{{color:#67e8f9;}}
.grid-2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.grid-3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.grid-4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}}
.grid-5{{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:9px}}
.metric{{background:#0d1420;border:1px solid var(--border);border-radius:10px;padding:12px 14px;text-align:center}}
.metric-lbl{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:5px}}
.metric-val{{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;word-break:break-all;}}
.metric-sub{{font-size:10px;color:var(--muted);margin-top:2px;word-wrap:break-word;}}
.match-card{{border-radius:10px;padding:10px 14px;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px}}
.match-icon{{font-size:18px;flex-shrink:0;margin-top:2px;}}
.match-label{{font-size:13px;font-weight:600;word-wrap:break-word;}}
.match-detail{{font-size:11px;color:var(--muted);margin-top:2px;word-wrap:break-word;word-break:break-all;}}
.match-ok{{background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2)}}
.match-near{{background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2)}}
.match-diff{{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2)}}
.mic-table,.ai-lab-table{{width:100%;border-collapse:collapse;table-layout:fixed;}}
.mic-table th,.ai-lab-table th{{padding:8px 10px;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border-bottom:1px solid var(--border);text-align:left;font-weight:600;word-break:break-word;}}
.mic-table td,.ai-lab-table td{{border-bottom:1px solid rgba(255,255,255,0.03);word-wrap:break-word;word-break:break-word;}}
.mic-table tr:hover td,.ai-lab-table tr:hover td{{background:rgba(255,255,255,0.02)}}
.chart-wrap{{position:relative;width:100%;height:210px}}
.chart-wrap-xl{{position:relative;width:100%;height:300px}}
.scope-viewport,.plate-viewport{{width:190px;height:190px;border-radius:50%;border:4px solid #334155;margin:0 auto;overflow:hidden}}
.scope-viewport{{background:#f5f0e8}}
.plate-viewport{{background:#1a2e1a;position:relative}}
.plate-label{{text-align:center;font-size:12px;color:var(--muted);margin-top:7px;word-wrap:break-word;}}
.info-row{{display:flex;justify-content:space-between;align-items:flex-start;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;gap:8px;flex-wrap:wrap;}}
.info-row:last-child{{border-bottom:none}}
.info-key{{color:var(--muted);flex-shrink:0;}}
.info-val{{font-weight:600;text-align:right;word-break:break-word;max-width:60%;}}
.log-box{{background:#060d1a;border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;max-height:210px;overflow-y:auto;line-height:1.8;word-break:break-all;}}
hr{{border:none;border-top:1px solid var(--border);margin:14px 0}}
</style>
</head>
<body>
<div class="lab-shell">

<div class="lab-header">
  <div style="display:grid;grid-template-columns:1fr auto;gap:16px;align-items:start;">
    <div>
      <div class="bac-name">&#128300; {sim['bacteria']}</div>
      <div class="bac-disease">{db['disease']}</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;">
        <span style="display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(0,201,255,0.1);color:#00c9ff;border:1px solid rgba(0,201,255,0.2)">Gram {db['gram']}</span>
        <span style="display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(124,58,237,0.1);color:#a78bfa;border:1px solid rgba(124,58,237,0.2)">{db['shape']}</span>
        <span style="display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.2)">{'Motile' if db['motility'] else 'Non-motile'}</span>
        <span style="display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(245,158,11,0.1);color:#fbbf24;border:1px solid rgba(245,158,11,0.2)">{'Spore-forming' if db['spore'] else 'Non-spore'}</span>
        <span style="display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(239,68,68,0.1);color:#f87171;border:1px solid rgba(239,68,68,0.2)">{'Encapsulated' if db['capsule'] else 'No capsule'}</span>
      </div>
    </div>
    <div style="text-align:right;font-size:12px;color:#e2e8f0;word-wrap:break-word;min-width:0;">
      <div style="font-size:10px;color:var(--muted);margin-bottom:3px;">LAB CONDITIONS</div>
      {sim['temperature']}&deg;C &middot; pH {sim['ph']} &middot; {sim['incubation_h']}h<br>
      10<sup>{sim['cfu_exp']}</sup> CFU/mL &middot; {sim['medium']}
    </div>
  </div>
</div>

<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('culture',this)">Culture</button>
  <button class="tab-btn" onclick="switchTab('mic',this)">MIC Plate</button>
  <button class="tab-btn" onclick="switchTab('ailab',this)">AI vs Lab</button>
  <button class="tab-btn" onclick="switchTab('compare',this)">Scores</button>
  <button class="tab-btn" onclick="switchTab('profile',this)">Drug Profile</button>
  <button class="tab-btn" onclick="switchTab('genes',this)">Genes</button>
</div>

<div id="tab-overview" class="tab-panel active">
  {alert_html}
  <div class="explain-box"><strong>&#9432; What this tab shows:</strong> High-level summary of the organism's resistance profile. MRI measures breadth of resistance, ARI reflects mechanism density, and AI values are compared against simulated lab measurements.</div>

  <div class="grid-5" style="margin-bottom:12px">
    <div class="metric"><div class="metric-lbl">Lab MRI</div><div class="metric-val" style="color:{risk_lab_col}">{sim['lab_mri']}</div><div class="metric-sub">{risk_lab_label}</div></div>
    <div class="metric"><div class="metric-lbl">AI MRI</div><div class="metric-val" style="color:{risk_ai_col}">{sim['ai_mri']}</div><div class="metric-sub">{risk_ai_label}</div></div>
    <div class="metric"><div class="metric-lbl">Lab ARI</div><div class="metric-val" style="color:#00c9ff">{sim['lab_ari']}</div><div class="metric-sub">Density index</div></div>
    <div class="metric"><div class="metric-lbl">Est. Genes</div><div class="metric-val">{sim['total_genes']}</div><div class="metric-sub">ARGs estimated</div></div>
    <div class="metric"><div class="metric-lbl">Drug Classes</div><div class="metric-val">{sim['u_drug_classes']}</div><div class="metric-sub">Classes resisted</div></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Organism identity</div>
      <div class="info-row"><span class="info-key">Gram stain</span><span class="info-val" style="color:{db['gram_color']}">{db['gram']}</span></div>
      <div class="info-row"><span class="info-key">Morphology</span><span class="info-val">{db['shape']}</span></div>
      <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
      <div class="info-row"><span class="info-key">Optimal temp</span><span class="info-val">{db['optimal_temp']}&deg;C</span></div>
      <div class="info-row"><span class="info-key">Optimal pH</span><span class="info-val">{db['optimal_ph']}</span></div>
      <div class="info-row"><span class="info-key">Oxidase</span><span class="info-val">{db['oxidase']}</span></div>
      <div class="info-row"><span class="info-key">Catalase</span><span class="info-val">{db['catalase']}</span></div>
      <div class="info-row"><span class="info-key">Lactose fermentation</span><span class="info-val">{'Positive' if db['ferments_lactose'] else 'Negative'}</span></div>
      <div class="info-row"><span class="info-key">Selective medium</span><span class="info-val">{db['selective_media']}</span></div>
    </div>
    <div class="card">
      <div class="card-title">Resistance profile (population benchmarks)</div>
      {res_bars}
      <div style="font-size:10px;color:var(--muted);margin-top:8px;line-height:1.6;">Based on curated species-level resistance data for <em>{sim['bacteria']}</em></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">AI vs Lab match summary</div>
    <div class="grid-3">
      <div class="match-card {'match-ok' if mri_delta<0.03 else 'match-near' if mri_delta<0.08 else 'match-diff'}">
        <span class="match-icon">{'&#10003;' if mri_delta<0.03 else '~' if mri_delta<0.08 else '&#10007;'}</span>
        <div><div class="match-label">MRI reading</div><div class="match-detail">Lab: {sim['lab_mri']} | AI: {sim['ai_mri']} | Δ={mri_delta}</div></div>
      </div>
      <div class="match-card {'match-ok' if ari_delta<0.02 else 'match-near' if ari_delta<0.06 else 'match-diff'}">
        <span class="match-icon">{'&#10003;' if ari_delta<0.02 else '~' if ari_delta<0.06 else '&#10007;'}</span>
        <div><div class="match-label">ARI reading</div><div class="match-detail">Lab: {sim['lab_ari']} | AI: {sim['ai_ari']} | Δ={ari_delta}</div></div>
      </div>
      <div class="match-card {'match-ok' if risk_agree else 'match-diff'}">
        <span class="match-icon">{'&#10003;' if risk_agree else '&#10007;'}</span>
        <div><div class="match-label">Risk classification</div><div class="match-detail">Lab: {risk_lab_label} | AI: {risk_ai_label}</div></div>
      </div>
    </div>
    <div style="margin-top:10px;padding:10px 14px;background:rgba(0,201,255,0.05);border-radius:8px;font-size:12px;color:#94a3b8;line-height:1.7;">
      Antibiotic concordance: <strong style="color:{'#22c55e' if match_pct>=80 else '#f59e0b' if match_pct>=60 else '#ef4444'}">{matched_ab}/{total_ab} agents ({match_pct}%)</strong> agree between lab and AI.
    </div>
  </div>
</div>

<div id="tab-culture" class="tab-panel">
  <div class="explain-box"><strong>&#9432; What this test shows:</strong> The growth curve tracks bacterial cell density over time, while colony morphology and Gram stain simulate common wet-lab identification steps.</div>
  <div class="grid-4" style="margin-bottom:12px">
    <div class="metric"><div class="metric-lbl">Growth rate</div><div class="metric-val" style="color:#00c9ff">{sim['growth_rate']}</div><div class="metric-sub">factor (1.0=optimal)</div></div>
    <div class="metric"><div class="metric-lbl">Final density</div><div class="metric-val">{round(sim['growth_cfu'][-1]/1e6,1)}</div><div class="metric-sub">&times;10&sup6; CFU/mL</div></div>
    <div class="metric"><div class="metric-lbl">Colonies</div><div class="metric-val">{sim['colony_count']}</div><div class="metric-sub">on plate</div></div>
    <div class="metric"><div class="metric-lbl">Gram stain</div><div class="metric-val" style="font-size:13px;color:{db['gram_color']}">{db['gram']}</div><div class="metric-sub">{db['shape']}</div></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Growth curve (CFU/mL &times; 10&sup6;)</div>
      <div class="chart-wrap"><canvas id="growthChart"></canvas></div>
      <div style="font-size:11px;color:var(--muted);margin-top:7px;">Medium: {sim['medium']} &middot; {sim['incubation_h']}h incubation</div>
    </div>
    <div class="card">
      <div class="card-title">Colony morphology plate</div>
      <div class="plate-viewport"><svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">{colony_svg}</svg></div>
      <div class="plate-label">{db['colony_size'].capitalize()} colonies &middot; {db['selective_media']}</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Gram stain microscopy (&#215;1000)</div>
    <div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap;">
      <div style="flex-shrink:0;">
        <div class="scope-viewport">
          <svg width="120" height="80" xmlns="http://www.w3.org/2000/svg" style="background:#f5f0e8">{gram_shape_svg}</svg>
        </div>
        <div class="plate-label">Gram {db['gram']} &middot; {db['shape']}</div>
      </div>
      <div style="flex:1;min-width:200px;">
        <div class="info-row"><span class="info-key">Gram stain result</span><span class="info-val" style="color:{db['gram_color']}">{'Purple (crystal violet)' if db['gram']=='Positive' else 'Pink/red (safranin)' if db['gram']=='Negative' else 'Red (acid-fast)'}</span></div>
        <div class="info-row"><span class="info-key">Cell morphology</span><span class="info-val">{db['shape']}</span></div>
        <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
        <div class="info-row"><span class="info-key">Spore formation</span><span class="info-val">{'Endospores visible' if db['spore'] else 'None'}</span></div>
        <div class="info-row"><span class="info-key">Capsule</span><span class="info-val">{'Present (India ink halo)' if db['capsule'] else 'Absent'}</span></div>
      </div>
    </div>
  </div>
</div>

<div id="tab-mic" class="tab-panel">
  <div class="explain-box"><strong>&#9432; What is MIC testing?</strong> Minimum Inhibitory Concentration is the lowest concentration preventing visible growth. The plate below simulates broth microdilution with doubling dilutions.</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">
    <span style="font-size:12px;color:var(--muted);flex-shrink:0;">Select antibiotic</span>
    <div id="ab-selector" style="display:flex;gap:5px;flex-wrap:wrap;"></div>
  </div>
  <div class="card">
    <div class="card-title" id="mic-plate-title">MIC Plate</div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:10px;">96-well format &middot; Rows replicates &middot; Columns doubling dilutions</div>
    <div id="mic-plate-grid" style="display:grid;grid-template-columns:repeat(12,1fr);gap:3px;margin-bottom:7px;"></div>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--muted);padding:0 2px;flex-wrap:wrap;gap:2px;">
      <span>0.06</span><span>0.12</span><span>0.25</span><span>0.5</span><span>1</span><span>2</span><span>4</span><span>8</span><span>16</span><span>32</span><span>64</span><span>128 µg/mL</span>
    </div>
    <div style="margin-top:12px;padding:9px 14px;background:#0d1420;border-radius:8px;font-size:13px;word-wrap:break-word;">
      MIC <strong id="mic-result-val" style="color:#00c9ff;font-family:'JetBrains Mono',monospace;">--</strong> µg/mL
      &nbsp;&nbsp; Interpretation <strong id="mic-result-interp">--</strong>
      &nbsp;&nbsp; S ≤ <span id="mic-s-break">--</span>
      &nbsp;&nbsp; R &gt; <span id="mic-r-break">--</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Complete MIC summary</div>
    <div style="overflow-x:auto;">
      <table class="mic-table">
        <thead>
          <tr><th>Antibiotic</th><th>Class</th><th>Lab MIC</th><th>Lab</th><th>AI MIC</th><th>AI</th><th style="text-align:center;">Match</th></tr>
        </thead>
        <tbody>{mic_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<div id="tab-ailab" class="tab-panel">
  <div class="explain-box"><strong>&#9432; AI Result vs Lab Result:</strong> Lab values come from simulated phenotypic testing, while AI values are genomic predictions. This tab compares them directly for concordance.</div>
  <div class="card">
    <div class="card-title">Full AI vs Lab comparison</div>
    <div style="overflow-x:auto;">
      <table class="ai-lab-table">
        <thead><tr><th style="width:30%">Test / Metric</th><th style="width:28%;color:#60a5fa;">Lab Result</th><th style="width:28%;color:#a78bfa;">AI Prediction</th><th style="width:14%;text-align:center;">Status</th></tr></thead>
        <tbody>{ai_lab_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card-title">MIC comparison chart (log₂ scaled)</div>
    <div class="chart-wrap-xl"><canvas id="micCompChart2"></canvas></div>
  </div>
</div>

<div id="tab-compare" class="tab-panel">
  <div class="explain-box"><strong>&#9432; What this shows:</strong> Side-by-side comparison of MRI and ARI scores plus an agreement log for all antibiotics tested.</div>
  <div class="grid-4" style="margin-bottom:12px">
    <div class="metric"><div class="metric-lbl">Lab MRI</div><div class="metric-val" style="color:{risk_lab_col}">{sim['lab_mri']}</div><div class="metric-sub">{risk_badge(risk_lab)}</div></div>
    <div class="metric"><div class="metric-lbl">AI MRI</div><div class="metric-val" style="color:{risk_ai_col}">{sim['ai_mri']}</div><div class="metric-sub">{risk_badge(risk_ai)}</div></div>
    <div class="metric"><div class="metric-lbl">MRI Delta</div><div class="metric-val" style="color:{'#22c55e' if mri_delta<0.03 else '#f59e0b' if mri_delta<0.08 else '#ef4444'}">{mri_delta}</div><div class="metric-sub">{mri_match}</div></div>
    <div class="metric"><div class="metric-lbl">AB Concordance</div><div class="metric-val" style="color:{'#22c55e' if match_pct>=80 else '#f59e0b' if match_pct>=60 else '#ef4444'}">{match_pct}</div><div class="metric-sub">{matched_ab}/{total_ab} agents</div></div>
  </div>
  <div class="grid-2">
    <div class="card"><div class="card-title">MRI / ARI bar comparison</div><div class="chart-wrap"><canvas id="mriChart"></canvas></div></div>
    <div class="card"><div class="card-title">Agreement log</div><div class="log-box" id="agreement-log"></div></div>
  </div>
</div>

<div id="tab-profile" class="tab-panel">
  <div class="explain-box"><strong>&#9432; What this shows:</strong> Drug class resistance radar with therapeutic candidate agents and resistant agents grouped by lab interpretation.</div>
  <div class="card"><div class="card-title">Drug class resistance radar</div><div class="chart-wrap-xl"><canvas id="profileChart"></canvas></div></div>
  <div class="card">
    <div class="card-title">Susceptibility zone analysis</div>
    <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:12px 14px;margin-bottom:10px;">
      <div style="color:#22c55e;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:8px;">Susceptible agents (S)</div>
      <div id="susc-list" style="display:flex;flex-wrap:wrap;gap:5px;"></div>
    </div>
    <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:12px 14px;">
      <div style="color:#f87171;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:8px;">Resistant agents (R)</div>
      <div id="resist-list" style="display:flex;flex-wrap:wrap;gap:5px;"></div>
    </div>
  </div>
</div>

<div id="tab-genes" class="tab-panel">
  <div class="explain-box"><strong>&#9432; What are resistance genes?</strong> ARGs encode mechanisms such as target alteration, efflux, degradation, or bypass that enable antibiotic resistance.</div>
  <div class="grid-3" style="margin-bottom:12px">
    <div class="metric"><div class="metric-lbl">Est. ARGs</div><div class="metric-val" style="color:#00c9ff">{sim['total_genes']}</div><div class="metric-sub">estimated for species</div></div>
    <div class="metric"><div class="metric-lbl">Drug classes</div><div class="metric-val">{sim['u_drug_classes']}</div><div class="metric-sub">unique classes</div></div>
    <div class="metric"><div class="metric-lbl">Mechanisms</div><div class="metric-val">{sim['u_mechs']}</div><div class="metric-sub">unique strategies</div></div>
  </div>
  <div class="card">
    <div class="card-title">Known resistance determinants</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">{gene_tags}</div>
    <div style="font-size:11px;color:var(--muted);margin-top:6px;line-height:1.6;">Curated reference list for this species.</div>
  </div>
  <div class="card"><div class="card-title">Associated diseases</div><div style="font-size:13px;line-height:1.8;color:#94a3b8;word-wrap:break-word;">{db['disease']}</div></div>
</div>

</div>

<script>
var SIMDATA = {{
  bacterianame: {json.dumps(sim["bacteria"])},
  growthtimes: {growth_labels_js},
  growthcfu: {growth_data_js},
  abnames: {json.dumps(ab_names)},
  labmics: {json.dumps([round(x,2) for x in lab_mics])},
  aimics: {json.dumps([round(x,2) for x in ai_mics])},
  classes: {json.dumps(classes)},
  labprofile: {json.dumps([round(x,2) for x in lab_profile_vals])},
  aiprofile: {json.dumps([round(x,2) for x in ai_profile_vals])},
  micresults: {json.dumps(sim["mic_results"])},
  labmri: {sim["lab_mri"]},
  aimri: {sim["ai_mri"]},
  labari: {sim["lab_ari"]},
  aiari: {sim["ai_ari"]},
  labrisk: {json.dumps(risk_lab_label)},
  airisk: {json.dumps(risk_ai_label)}
}};

function switchTab(name, btn){{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}

new Chart(document.getElementById('growthChart'), {{
  type: 'line',
  data: {{
    labels: SIMDATA.growthtimes,
    datasets: [{{label:'CFU/mL x10^6', data:SIMDATA.growthcfu, borderColor:'#00c9ff', backgroundColor:'rgba(0,201,255,0.06)', fill:true, tension:0.4, pointRadius:2}}]
  }},
  options: {{responsive:true, maintainAspectRatio:false, plugins:{{legend:{{display:false}}}}}}
}});

new Chart(document.getElementById('mriChart'), {{
  type:'bar',
  data: {{
    labels:['MRI Score','ARI Score'],
    datasets:[
      {{label:'Lab', data:[SIMDATA.labmri,SIMDATA.labari], backgroundColor:'#185FA5', borderRadius:4}},
      {{label:'AI', data:[SIMDATA.aimri,SIMDATA.aiari], backgroundColor:'rgba(0,201,255,0.3)', borderColor:'#00c9ff', borderWidth:2, borderRadius:4}}
    ]
  }},
  options: {{responsive:true, maintainAspectRatio:false, scales:{{y:{{min:0,max:1}}}}}}
}});

var micCompCtx2 = document.getElementById('micCompChart2');
if (micCompCtx2) {{
  new Chart(micCompCtx2, {{
    type:'bar',
    data: {{
      labels:SIMDATA.abnames,
      datasets:[
        {{label:'Lab MIC log₂', data:SIMDATA.labmics, backgroundColor:'#185FA5', borderRadius:3}},
        {{label:'AI MIC log₂', data:SIMDATA.aimics, backgroundColor:'rgba(0,201,255,0.3)', borderColor:'#00c9ff', borderWidth:1, borderRadius:3}}
      ]
    }},
    options: {{responsive:true, maintainAspectRatio:false}}
  }});
}}

new Chart(document.getElementById('profileChart'), {{
  type:'radar',
  data: {{
    labels:SIMDATA.classes,
    datasets:[
      {{label:'Lab', data:SIMDATA.labprofile, borderColor:'#185FA5', backgroundColor:'rgba(24,95,165,0.15)', borderWidth:2}},
      {{label:'AI', data:SIMDATA.aiprofile, borderColor:'#00c9ff', backgroundColor:'rgba(0,201,255,0.08)', borderWidth:2, borderDash:[5,3]}}
    ]
  }},
  options: {{responsive:true, maintainAspectRatio:false}}
}});

var DILUTIONSARR = [0.06,0.12,0.25,0.5,1,2,4,8,16,32,64,128];

function buildAbSelector(){{
  var sel = document.getElementById('ab-selector');
  SIMDATA.micresults.forEach(function(ab,i){{
    var btn = document.createElement('button');
    btn.textContent = ab.name;
    btn.style.cssText = "padding:3px 9px;font-size:11px;border-radius:4px;cursor:pointer;border:1px solid rgba(255,255,255,0.12);background:" + (i===0 ? "rgba(0,201,255,0.15)" : "rgba(255,255,255,0.04)") + ";color:" + (i===0 ? "#00c9ff" : "#94a3b8") + ";font-family:Sora,sans-serif;white-space:nowrap;";
    btn.onclick = function(){{
      document.querySelectorAll('#ab-selector button').forEach(function(b,j){{
        b.style.background = j===i ? "rgba(0,201,255,0.15)" : "rgba(255,255,255,0.04)";
        b.style.color = j===i ? "#00c9ff" : "#94a3b8";
      }});
      renderMicPlate(i);
    }};
    sel.appendChild(btn);
  }});
}}

function renderMicPlate(idx){{
  var ab = SIMDATA.micresults[idx], mic = ab.lab_mic;
  var micColIdx = DILUTIONSARR.indexOf(DILUTIONSARR.reduce(function(a,b){{ return Math.abs(b-mic) < Math.abs(a-mic) ? b : a; }}));
  var grid = document.getElementById('mic-plate-grid');
  grid.innerHTML = "";
  for (var row=0; row<8; row++) {{
    for (var col=0; col<12; col++) {{
      var well = document.createElement('div');
      well.style.cssText = "aspect-ratio:1;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:7px;font-weight:700;";
      if (col === micColIdx) {{
        well.style.background = "#1d4ed8";
        well.style.border = "2px solid #60a5fa";
        well.style.color = "#fff";
        well.textContent = "MIC";
      }} else if (col < micColIdx) {{
        well.style.background = "rgba(239,68,68,0.6)";
        well.style.border = "1px solid #ef4444";
        well.style.color = "#fff";
      }} else {{
        well.style.background = "rgba(34,197,94,0.2)";
        well.style.border = "1px solid rgba(34,197,94,0.4)";
        well.style.color = "#4ade80";
        well.textContent = "-";
      }}
      grid.appendChild(well);
    }}
  }}
  document.getElementById('mic-result-val').textContent = mic;
  var iC = {{S:"#22c55e", I:"#f59e0b", R:"#ef4444"}}[ab.lab_interp];
  var iT = {{S:"Susceptible", I:"Intermediate", R:"Resistant"}}[ab.lab_interp];
  document.getElementById('mic-result-interp').innerHTML = '<span style="color:'+iC+';font-weight:700;">'+iT+'</span>';
  document.getElementById('mic-s-break').textContent = ab.S_break + " µg/mL";
  document.getElementById('mic-r-break').textContent = ab.R_break + " µg/mL";
  document.getElementById('mic-plate-title').textContent = ab.name + " — " + ab.class;
}}

function buildAgreementLog(){{
  var log = document.getElementById('agreement-log'), lines = [];
  var delta = Math.abs(SIMDATA.labmri - SIMDATA.aimri);
  lines.push({{t:'ok', m:'LAB MRI  ' + SIMDATA.labmri + '  ' + SIMDATA.labrisk}});
  lines.push({{t:'info', m:'AI MRI   ' + SIMDATA.aimri + '  ' + SIMDATA.airisk}});
  lines.push({{t: delta<0.03 ? 'ok' : delta<0.08 ? 'warn' : 'err', m:'CMP Δ    ' + delta.toFixed(3)}});
  lines.push({{t:'ok', m:'LAB ARI  ' + SIMDATA.labari}});
  lines.push({{t:'info', m:'AI ARI   ' + SIMDATA.aiari}});
  SIMDATA.micresults.forEach(function(r){{
    var m = r.lab_interp === r.ai_interp;
    lines.push({{t: m ? 'ok' : 'warn', m:'MIC ' + r.name + '  Lab:' + r.lab_mic + '(' + r.lab_interp + ')  AI:' + r.ai_mic + '(' + r.ai_interp + ')'}});
  }});
  log.innerHTML = lines.map(function(l){{
    var c = l.t==='ok' ? '#22c55e' : l.t==='info' ? '#00c9ff' : l.t==='warn' ? '#f59e0b' : '#ef4444';
    return '<div style="color:'+c+';">'+l.m+'</div>';
  }}).join('');
}}

function buildSuscZone(){{
  var sl = document.getElementById('susc-list'), rl = document.getElementById('resist-list');
  SIMDATA.micresults.forEach(function(r){{
    var tag = document.createElement('span');
    tag.style.cssText = "display:inline-block;padding:3px 9px;border-radius:20px;font-size:12px;font-weight:600;margin:2px;";
    tag.textContent = r.name;
    if (r.lab_interp === 'S') {{
      tag.style.background = "rgba(34,197,94,0.1)";
      tag.style.color = "#4ade80";
      tag.style.border = "1px solid rgba(34,197,94,0.25)";
      sl.appendChild(tag);
    }} else if (r.lab_interp === 'R') {{
      tag.style.background = "rgba(239,68,68,0.1)";
      tag.style.color = "#f87171";
      tag.style.border = "1px solid rgba(239,68,68,0.25)";
      rl.appendChild(tag);
    }}
  }});
  if (!sl.children.length) sl.innerHTML = '<span style="color:#64748b;font-size:12px;">None found</span>';
  if (!rl.children.length) rl.innerHTML = '<span style="color:#64748b;font-size:12px;">None found</span>';
}}

buildAbSelector();
renderMicPlate(0);
buildAgreementLog();
buildSuscZone();
</script>
</body>
</html>
"""
    return html


# -------------------------------------------------------------------
# ENHANCED STREAMLIT LAB TAB (REPLACE OLD renderlabtab)
# -------------------------------------------------------------------
def renderlabtab():
    st.markdown("""
    <style>
    .lab-section-title{
      font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
      color:#00d4ff;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid rgba(0,212,255,.15);
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown div {
      word-wrap:break-word;overflow-wrap:break-word;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0c1f3f,#111827);border:1px solid rgba(0,212,255,.15);
                border-radius:14px;padding:18px 22px;margin-bottom:18px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,#00d4ff,#7c3aed,#10b981,#00d4ff);
                  background-size:300%;animation:scanBar 4s linear infinite;"></div>
      <style>@keyframes scanBar{0%{background-position:0%}100%{background-position:300%}}</style>
      <h2 style="color:#00d4ff;font-size:1.2rem;margin-bottom:6px;">&#129516; Virtual Laboratory Simulation</h2>
      <p style="color:#64748b;font-size:13px;margin:0;line-height:1.6;word-wrap:break-word;">
        Full phenotypic simulation — culture curves, MIC plates, Gram stain microscopy, and AI vs Lab comparison.
        All results are specific to the <strong style="color:#00d4ff;">selected organism</strong>.
        Run the simulation, then download a comprehensive <strong style="color:#00d4ff;">Lab Report PDF</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_bac, col_preset = st.columns([2, 2])
    with col_bac:
        st.markdown('<div class="lab-section-title">Select Organism</div>', unsafe_allow_html=True)
        all_bac_names = sorted(LABBACTERIADB.keys())
        selected_bac = st.selectbox("Bacterial species", all_bac_names, label_visibility="collapsed", key="lab_bac_select")

    with col_preset:
        st.markdown('<div class="lab-section-title">Quick Presets</div>', unsafe_allow_html=True)
        preset = st.selectbox(
            "Quick presets",
            ["-- custom --", "MRSA (High resistance)", "ESBL E. coli", "XDR Acinetobacter",
             "Pan-susceptible Salmonella", "MDR M. tuberculosis", "VRE Enterococcus"],
            label_visibility="collapsed",
            key="lab_preset"
        )

    preset_map = {
        "MRSA (High resistance)": {"bac": "Staphylococcus aureus", "res": 0.85},
        "ESBL E. coli": {"bac": "Escherichia coli", "res": 0.72},
        "XDR Acinetobacter": {"bac": "Acinetobacter baumannii", "res": 0.92},
        "Pan-susceptible Salmonella": {"bac": "Salmonella enterica", "res": 0.10},
        "MDR M. tuberculosis": {"bac": "Mycobacterium tuberculosis", "res": 0.78},
        "VRE Enterococcus": {"bac": "Enterococcus faecium", "res": 0.80},
    }

    resistance_override = 0.5
    if preset in preset_map:
        selected_bac = preset_map[preset]["bac"]
        resistance_override = preset_map[preset]["res"]

    st.markdown("---")
    db = LABBACTERIADB[selected_bac]
    st.markdown(
        f'<div class="lab-section-title">Lab Conditions — <em style="text-transform:none;">{selected_bac}</em> '
        f'(Optimal: {db["optimal_temp"]}°C, pH {db["optimal_ph"]})</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        temperature = st.slider("Temperature (°C)", 25.0, 45.0, float(db["optimal_temp"]), 0.5, key="lab_temp")
    with c2:
        ph = st.slider("pH", 5.5, 9.0, float(db["optimal_ph"]), 0.1, key="lab_ph")
    with c3:
        incubation_h = st.slider("Incubation (h)", 4, 72, 18, 1, key="lab_inc")
    with c4:
        cfu_exp = st.slider("Inoculum 10^x CFU/mL", 3, 8, 5, 1, key="lab_cfu")
    with c5:
        if preset not in preset_map:
            resistance_override = st.slider("Resistance pressure", 0.0, 1.0, 0.5, 0.05, key="lab_res")
        else:
            st.metric("Resistance pressure", f"{resistance_override:.2f}", delta="preset")

    medium = st.selectbox(
        "Growth medium",
        ["Mueller-Hinton Broth (MHB)", "Lysogeny Broth Agar (LBA)",
         "Columbia Blood Agar (CBA)", "Brain Heart Infusion (BHI)",
         "Lowenstein-Jensen (LJ)", "Chocolate Agar"],
        key="lab_medium"
    )

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True, key="lab_run_btn")

    if "lab_result_html" not in st.session_state:
        st.session_state.lab_result_html = None
        st.session_state.lab_run_key = None
        st.session_state.lab_sim_data = None

    run_key = f"{selected_bac}|{temperature}|{ph}|{incubation_h}|{cfu_exp}|{resistance_override}|{medium}"

    if run_btn or (st.session_state.lab_run_key == run_key and st.session_state.lab_result_html):
        if run_btn or st.session_state.lab_result_html is None:
            with st.spinner(f"Running virtual lab for {selected_bac}..."):
                sim = labsimulate(selected_bac, resistance_override, temperature, ph, incubation_h, cfu_exp, medium)
                html = labbuildhtml(sim)
                st.session_state.lab_result_html = html
                st.session_state.lab_run_key = run_key
                st.session_state.lab_sim_data = sim

        components.html(st.session_state.lab_result_html, height=1000, scrolling=True)

        st.markdown("---")
        st.markdown("### 📄 Download Comprehensive Lab Report")

        with st.expander("ℹ️ What's included in the Lab Report PDF?", expanded=False):
            st.markdown("""
1. Cover page with organism and lab conditions  
2. Organism identity and biochemical profile  
3. MRI and ARI score summary  
4. Growth kinetics overview  
5. MIC results table  
6. Resistance gene section  
7. AI vs Lab summary  
""")

        col_dl, col_info = st.columns([1, 2])

        with col_dl:
            if st.button("Generate Lab Report PDF", type="secondary", use_container_width=True, key="lab_pdf_btn"):
                with st.spinner("Generating comprehensive lab report..."):
                    try:
                        simdata = st.session_state.lab_sim_data
                        pdfbytes = generatelabreportpdf(simdata)
                        safe_name = simdata["bacteria"].replace(" ", "_")
                        st.download_button(
                            label=f"⬇ Download {simdata['bacteria']} Lab Report",
                            data=pdfbytes,
                            filename=f"VirtualLab_{safe_name}_Report.pdf",
                            mime="application/pdf",
                            key="lab_pdf_download"
                        )
                        st.success("Lab report ready. Click the download button.")
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")

        with col_info:
            if st.session_state.lab_sim_data:
                s = st.session_state.lab_sim_data
                st.info(
                    f"Current simulation: {s['bacteria']} | MRI: {s['lab_mri']} | "
                    f"Risk: {s['lab_risk'][0]} | Antibiotics tested: {len(s['mic_results'])}"
                )
    else:
        st.markdown(
            f"""
            <div style="text-align:center;padding:55px 20px;background:rgba(15,23,42,0.5);
                        border:1px dashed rgba(0,212,255,.2);border-radius:14px;margin-top:10px;">
              <div style="font-size:3rem;margin-bottom:12px;">&#129516;</div>
              <div style="font-size:17px;color:#e2e8f0;font-weight:600;margin-bottom:8px;">Virtual Lab Ready</div>
              <div style="font-size:13px;color:#64748b;line-height:1.7;word-wrap:break-word;">
                Selected <strong style="color:#00d4ff;">{selected_bac}</strong><br>
                Configure lab conditions above, then click <strong style="color:#00d4ff;">Run Simulation</strong>.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
