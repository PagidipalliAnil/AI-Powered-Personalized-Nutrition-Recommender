import streamlit as st
import pandas as pd
import os, json, re, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

from utils.bmi import calculate_bmi, bmi_category, ideal_weight_range, daily_calorie_needs
from utils.ocr_reader import extract_text
from utils.condition_detector import detect_condition
from utils.recommender import (
    load_food_dataset, load_recipe_dataset,
    recommend_food, recommend_food_for_goal,
    GOAL_CRITERIA, SKINCARE_GOALS, CAT_EMOJI, CONDITION_CRITERIA,
)

st.set_page_config(
    page_title="NutriAI — AI Nutrition Recommender",
    page_icon="🥗", layout="wide",
    initial_sidebar_state="expanded",
)

USER_FILE = "users.json"
LOG_FILE  = "calories_log.json"
for fp, default in [(USER_FILE, {}), (LOG_FILE, [])]:
    if not os.path.exists(fp):
        with open(fp, "w") as fh: json.dump(default, fh)

# ─── State helpers ────────────────────────────────────────────────────────────
def ss(key, default=None):
    """Safe session_state getter."""
    return st.session_state.get(key, default)

def _persist(key, val):
    """Persist a value in session_state, never overwriting with None/empty."""
    if val is not None:
        st.session_state[key] = val

# ─── JSON helpers ─────────────────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return {} if "users" in path else []

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def log_food(user, food, cal, prot, fat, carbs, fiber=0, sodium=0):
    data = load_json(LOG_FILE)
    data.append({"user":user,"food":food,
                 "calories":float(cal),"protein":float(prot),
                 "fat":float(fat),"carbs":float(carbs),
                 "fiber":float(fiber),"sodium":float(sodium),
                 "datetime":str(datetime.now()),"date":str(datetime.now().date())})
    save_json(LOG_FILE, data)

def get_user_logs(user):
    return [d for d in load_json(LOG_FILE) if d.get("user") == user]

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#020617,#0f172a,#1e1b4b);color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(15,23,42,0.97)!important;border-right:1px solid #334155;}
h1,h2,h3{color:#f8fafc;font-weight:700;}

/* ── Food Cards ─────────────────────────────────────────────────── */
.fc-grid{display:grid;gap:18px;margin-bottom:8px;}
.fc{background:linear-gradient(160deg,#1e293b 0%,#0f172a 100%);
    border:1px solid #2d3f55;border-radius:18px;overflow:hidden;
    display:flex;flex-direction:column;
    box-shadow:0 2px 12px rgba(0,0,0,.35);
    transition:box-shadow .25s,transform .25s,border-color .25s;}
.fc:hover{box-shadow:0 10px 36px rgba(99,102,241,.32);
          transform:translateY(-4px);border-color:#4f46e5;}
.fc-img-wrap{position:relative;width:100%;height:180px;overflow:hidden;background:#0a1628;}
.fc-img{width:100%;height:180px;object-fit:cover;display:block;}
.fc-cat-strip{position:absolute;top:0;left:0;right:0;
  background:linear-gradient(180deg,rgba(0,0,0,.55) 0%,transparent 100%);
  padding:10px 12px 20px;}
.fc-cat-pill{display:inline-block;font-size:.62rem;font-weight:800;
  padding:3px 10px;border-radius:999px;letter-spacing:.06em;
  backdrop-filter:blur(4px);border:1px solid rgba(255,255,255,.12);}
.fc-ds-pill{display:inline-block;font-size:.58rem;font-weight:700;
  padding:2px 7px;border-radius:4px;background:#064e3b;
  color:#6ee7b7;margin-left:6px;vertical-align:middle;}
.fc-body{padding:14px 16px 6px;flex:1;display:flex;flex-direction:column;gap:0;}
.fc-title{font-size:1rem;font-weight:800;color:#f1f5f9;text-align:center;
          line-height:1.3;margin-bottom:10px;letter-spacing:-.01em;}
.fc-macros{background:#0a1628;border-radius:10px;padding:10px 12px;margin-bottom:10px;}
.macro-row{display:flex;justify-content:space-between;align-items:center;
           padding:4px 0;border-bottom:1px solid rgba(51,65,85,.4);}
.macro-row:last-child{border-bottom:none;}
.macro-row span{color:#64748b;font-size:.76rem;font-weight:500;}
.macro-row b{color:#e2e8f0;font-weight:700;font-size:.76rem;}
.macro-highlight{color:#a5b4fc!important;}
/* Recipe accordion */
.recipe-details{margin-top:4px;}
.recipe-details summary{cursor:pointer;list-style:none;
  display:flex;align-items:center;gap:8px;
  padding:8px 12px;background:#0a1628;border:1px solid #2d3f55;
  border-radius:10px;color:#818cf8;font-size:.78rem;font-weight:700;
  user-select:none;transition:background .18s;}
.recipe-details summary::before{content:"📖";font-size:.85rem;}
.recipe-details summary:hover{background:#172033;border-color:#4f46e5;}
.recipe-details[open] summary{border-radius:10px 10px 0 0;border-bottom:none;
  background:#172033;}
.recipe-content{background:#0a1628;border:1px solid #2d3f55;border-top:none;
  border-radius:0 0 10px 10px;padding:14px 14px 12px;}
.recipe-name-tag{font-size:.72rem;font-weight:800;color:#fbbf24;
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;}
.recipe-ings-wrap{margin-bottom:10px;}
.recipe-ings-label{font-size:.72rem;font-weight:700;color:#94a3b8;margin-bottom:6px;}
.recipe-ings-grid{display:flex;flex-wrap:wrap;gap:4px;}
.recipe-ing-chip{background:#1e293b;border:1px solid #334155;border-radius:6px;
  padding:3px 9px;font-size:.71rem;color:#94a3b8;}
.recipe-steps-label{font-size:.72rem;font-weight:700;color:#34d399;margin-bottom:6px;}
.recipe-steps-text{font-size:.78rem;color:#94a3b8;line-height:1.65;margin:0;}

/* ── OCR Health Report Panel ─────────────────────────────────────── */
.rp-wrap{background:#080f1e;border:1px solid #1e3a5f;border-radius:18px;
  overflow:hidden;margin:12px 0;}
.rp-header{background:linear-gradient(135deg,#0c1a3a,#111827);
  border-bottom:1px solid #1e3a5f;padding:16px 22px;
  display:flex;align-items:center;justify-content:space-between;}
.rp-header-left{display:flex;align-items:center;gap:12px;}
.rp-header-icon{font-size:1.6rem;}
.rp-header-title{font-size:1rem;font-weight:800;color:#f1f5f9;margin:0;}
.rp-header-sub{font-size:.73rem;color:#64748b;margin-top:2px;}
.rp-summary-pills{display:flex;gap:8px;flex-wrap:wrap;}
.rp-pill{display:inline-flex;align-items:center;gap:5px;
  padding:4px 12px;border-radius:999px;font-size:.72rem;font-weight:700;}
.rp-pill-red{background:rgba(248,113,113,.15);color:#f87171;border:1px solid rgba(248,113,113,.3);}
.rp-pill-yellow{background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid rgba(251,191,36,.3);}
.rp-pill-green{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3);}
.rp-pill-blue{background:rgba(129,140,248,.15);color:#818cf8;border:1px solid rgba(129,140,248,.3);}
.rp-body{padding:18px 22px;}
/* Lab value groups */
.rp-group{margin-bottom:20px;}
.rp-group-title{font-size:.7rem;font-weight:800;letter-spacing:.1em;
  text-transform:uppercase;color:#475569;margin-bottom:10px;
  display:flex;align-items:center;gap:8px;}
.rp-group-title::after{content:"";flex:1;height:1px;background:#1e293b;}
.rp-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;}
/* Individual metric card */
.rp-card{background:#111827;border-radius:12px;padding:12px 14px;
  position:relative;overflow:hidden;border:1px solid #1e293b;}
.rp-card::before{content:"";position:absolute;top:0;left:0;bottom:0;width:4px;border-radius:2px 0 0 2px;}
.rp-card.rp-normal::before{background:#34d399;}
.rp-card.rp-borderline::before{background:#fbbf24;}
.rp-card.rp-abnormal::before{background:#f87171;}
.rp-card.rp-info::before{background:#818cf8;}
.rp-card.rp-normal{border-color:rgba(52,211,153,.15);}
.rp-card.rp-borderline{border-color:rgba(251,191,36,.18);}
.rp-card.rp-abnormal{border-color:rgba(248,113,113,.2);}
.rp-metric-name{font-size:.68rem;font-weight:700;color:#64748b;
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;}
.rp-metric-value{font-size:1.35rem;font-weight:800;color:#f1f5f9;
  line-height:1;margin-bottom:4px;}
.rp-metric-unit{font-size:.72rem;font-weight:500;color:#475569;margin-left:3px;}
.rp-metric-status{display:inline-flex;align-items:center;gap:4px;
  font-size:.68rem;font-weight:700;padding:2px 8px;border-radius:999px;margin-top:4px;}
.rp-status-normal{background:rgba(52,211,153,.12);color:#34d399;}
.rp-status-borderline{background:rgba(251,191,36,.12);color:#fbbf24;}
.rp-status-abnormal{background:rgba(248,113,113,.12);color:#f87171;}
.rp-status-info{background:rgba(129,140,248,.12);color:#818cf8;}
/* Observations */
.rp-obs-section{margin-top:4px;}
.rp-obs-title{font-size:.7rem;font-weight:800;letter-spacing:.1em;
  text-transform:uppercase;color:#475569;margin-bottom:10px;
  display:flex;align-items:center;gap:8px;}
.rp-obs-title::after{content:"";flex:1;height:1px;background:#1e293b;}
.rp-obs-item{display:flex;gap:12px;align-items:flex-start;
  background:#111827;border-radius:10px;padding:10px 14px;
  margin-bottom:8px;border:1px solid #1e293b;}
.rp-obs-num{min-width:22px;height:22px;background:#1e293b;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:.65rem;font-weight:800;color:#818cf8;flex-shrink:0;margin-top:1px;}
.rp-obs-text{font-size:.81rem;color:#94a3b8;line-height:1.6;margin:0;}
/* Diagnoses banner */
.rp-diag-banner{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.25);
  border-radius:12px;padding:12px 16px;margin-bottom:16px;
  display:flex;align-items:flex-start;gap:12px;}
.rp-diag-icon{font-size:1.3rem;flex-shrink:0;}
.rp-diag-content{flex:1;}
.rp-diag-title{font-size:.72rem;font-weight:800;color:#f87171;
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;}
.rp-diag-pills{display:flex;flex-wrap:wrap;gap:6px;}
.rp-diag-pill{background:rgba(248,113,113,.15);color:#fca5a5;
  border:1px solid rgba(248,113,113,.3);border-radius:6px;
  padding:3px 10px;font-size:.75rem;font-weight:700;}

/* Metrics */
.metric-card{background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #334155;
  border-radius:14px;padding:16px;text-align:center;margin-bottom:10px;}
.metric-val{font-size:1.8rem;font-weight:800;color:#818cf8;}
.metric-lbl{font-size:.77rem;color:#64748b;margin-top:3px;}
.badge{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;font-size:.83rem;margin:2px;}
.badge-green{background:#064e3b;color:#6ee7b7;}
.badge-yellow{background:#451a03;color:#fcd34d;}
.badge-red{background:#450a0a;color:#fca5a5;}
.badge-blue{background:#0c1a4a;color:#93c5fd;}

/* Wellness goal */
.goal-hero{background:linear-gradient(135deg,#1e1b4b,#0f172a);
  border:1px solid #4f46e5;border-radius:16px;padding:20px 24px;margin-bottom:16px;}
.goal-stat{background:#0f172a;border:1px solid #1e293b;border-radius:10px;
  padding:10px 14px;text-align:center;}
.goal-stat .sv{font-size:1.4rem;font-weight:800;color:#818cf8;}
.goal-stat .sl{font-size:.72rem;color:#64748b;margin-top:2px;}
.tip-card{background:#1e293b;border-left:3px solid #818cf8;border-radius:0 10px 10px 0;
  padding:10px 14px;margin:6px 0;font-size:.81rem;color:#94a3b8;}

/* Skin care */
.skin-card{background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #334155;
  border-radius:16px;overflow:hidden;margin-bottom:12px;}
.skin-card-header{padding:14px 18px 10px;border-bottom:1px solid #334155;}
.skin-recipe{background:#0f172a;border:1px solid #334155;border-radius:10px;
  padding:14px;margin-top:10px;}
.skin-ingredient{display:inline-block;background:#1e293b;border:1px solid #334155;
  border-radius:6px;padding:3px 10px;font-size:.74rem;color:#94a3b8;margin:2px;}

/* Charts */
.chart-box{background:#1e293b;border:1px solid #334155;border-radius:14px;
  padding:16px;margin:8px 0;}
.stTabs [data-baseweb="tab"]{color:#94a3b8;font-weight:600;}
.stTabs [aria-selected="true"]{color:#818cf8!important;border-bottom-color:#818cf8!important;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;
  border:none;border-radius:10px;font-weight:600;padding:8px 20px;transition:opacity .2s;}
.stButton>button:hover{opacity:.82;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def condition_badge(cond):
    css = {"Normal":"badge-green","Underweight":"badge-blue",
           "Pre-Diabetes":"badge-yellow","Diabetes":"badge-red",
           "Pre-Hypertension":"badge-yellow","Hypertension":"badge-red",
           "Heart Risk":"badge-red","Borderline Cholesterol":"badge-yellow",
           "Obesity":"badge-yellow","Anemia":"badge-yellow"}.get(cond,"badge-blue")
    return f'<span class="badge {css}">{cond}</span>'

@st.cache_data(show_spinner=False)
def get_datasets():
    return load_food_dataset(), load_recipe_dataset()

# ══════════════════════════════════════════════════════════════════════════════
#  PROFESSIONAL OCR HEALTH REPORT PANEL
# ══════════════════════════════════════════════════════════════════════════════

# Medical groupings — label keywords → group name
_METRIC_GROUPS = {
    "Blood Glucose & Diabetes": [
        "glucose","fasting blood","blood sugar","fbs","hba1c","hba","glycated",
        "insulin","fructosamine","ppbs","postprandial",
    ],
    "Blood Pressure & Cardiac": [
        "blood pressure","systolic","diastolic","bp ","heart rate","pulse",
    ],
    "Lipid Panel": [
        "cholesterol","ldl","hdl","triglyceride","vldl","non-hdl","lipoprotein",
    ],
    "Blood Count": [
        "hemoglobin","haemoglobin","hb ","wbc","rbc","platelets","hematocrit",
        "mcv","mch","mchc","neutrophil","lymphocyte","eosinophil","basophil",
        "monocyte","white blood","red blood","blood count","cbc",
    ],
    "Iron & Anemia": [
        "iron","ferritin","tibc","transferrin","serum iron","iron saturation",
    ],
    "Kidney Function": [
        "creatinine","urea","bun","egfr","gfr","uric acid","albumin",
        "creatinine clearance","kidney",
    ],
    "Liver Function": [
        "sgot","sgpt","ast","alt","bilirubin","alkaline phosphatase","alp",
        "gamma gt","ggt","protein","albumin","liver",
    ],
    "Thyroid": [
        "tsh","t3","t4","free t3","free t4","thyroid","thyroxine","triiodothyronine",
    ],
    "Vitamins & Minerals": [
        "vitamin d","vitamin b12","vitamin b","vit d","vit b","calcium","phosphorus",
        "magnesium","sodium","potassium","chloride","zinc","folate","folic",
    ],
    "Diagnoses & Conditions": [
        "diagnosis","diagnoses","detected","condition","disease","disorder",
        "diabetes","hypertension","anemia","anaemia","hypothyroid","hyperthyroid",
        "early type","intolerance","deficiency","risk",
    ],
    "Observations & Notes": [
        "observation","family history","history","recommend","test mentioned",
        "impression","note","finding","glucose levels","insulin release",
        "gastric","glycated hemoglobin","fructosamine","microalbumin",
    ],
}

def _classify_line(label: str) -> str:
    """Return the group name for a given metric label."""
    l = label.lower()
    for group, keywords in _METRIC_GROUPS.items():
        if any(k in l for k in keywords):
            return group
    return "Other Findings"

def _parse_status(status_raw: str, label: str) -> tuple:
    """Return (display_text, css_class, pill_class) from raw status string."""
    s = status_raw.strip()
    sl = s.lower()
    # Strip emojis for clean display text
    clean = re.sub(r'[✅⚠️🔴ℹ️]', '', s).strip(" —–").strip()
    if not clean:
        clean = "Normal"

    if any(x in sl for x in ["🔴","abnormal","high","low","detected","critical","elevated","deficient"]):
        return clean, "rp-abnormal", "rp-status-abnormal", "rp-pill-red"
    elif any(x in sl for x in ["⚠️","borderline","moderate","mild"]):
        return clean, "rp-borderline", "rp-status-borderline", "rp-pill-yellow"
    elif any(x in sl for x in ["✅","normal","within range","optimal"]) or sl in ("", "normal"):
        return clean, "rp-normal", "rp-status-normal", "rp-pill-green"
    else:
        return clean, "rp-info", "rp-status-info", "rp-pill-blue"

def _split_value_unit(raw_value: str) -> tuple:
    """Split '148.0 mg/dL' → ('148.0', 'mg/dL')."""
    m = re.match(r'^([\d./]+)\s*(.*)$', raw_value.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw_value.strip(), ""

def render_ocr_panel(text_lines: list):
    """Render a professional clinical-grade health report panel."""
    import html as _h

    if not text_lines:
        st.info("No health data extracted. Try a clearer image or use manual entry.")
        return

    # ── Parse all lines ───────────────────────────────────────────────────────
    numeric_pat = re.compile(
        r'^(.+?):\s*([\d./]+(?:\s*/\s*[\d./]+)?\s*(?:[a-zA-Z%µ·/]+)?(?:\s*[a-zA-Z%µ·/]+)?)\s*(?:[—–]\s*(.+))?$'
    )
    obs_pat = re.compile(
        r'^(?:Observation|Test Mentioned|Diagnosis|Diagnoses):\s*(.+?)(?:\s*[—–]\s*(.+))?$',
        re.IGNORECASE
    )

    metrics     = []   # (label, value, unit, status_text, card_cls, pill_cls)
    diagnoses   = []   # detected condition names
    observations = []  # free-text clinical notes

    for line in text_lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Diagnosis line
        if re.match(r'^Diagnosis(?:es)?:', line, re.IGNORECASE):
            m = obs_pat.match(line)
            content = m.group(1).strip() if m else line.split(":",1)[1].strip()
            # Strip status part
            content = re.sub(r'\s*[—–].*$', '', content).strip()
            content = re.sub(r'[🔴⚠️✅ℹ️]','', content).strip()
            if content:
                diagnoses.append(content)
            continue

        # Observation / Test Mentioned line
        if re.match(r'^(?:Observation|Test Mentioned):', line, re.IGNORECASE):
            m = obs_pat.match(line)
            content = m.group(1).strip() if m else line.split(":",1)[1].strip()
            content = re.sub(r'[🔴⚠️✅ℹ️]','', content).strip()
            if content:
                observations.append(content)
            continue

        # Numeric metric line: "Label: Value Unit — Status"
        m = numeric_pat.match(line)
        if m:
            label  = m.group(1).strip()
            raw_v  = m.group(2).strip()
            status = (m.group(3) or "").strip()
            val, unit = _split_value_unit(raw_v)
            disp_status, card_cls, status_cls, pill_cls = _parse_status(status, label)
            metrics.append((label, val, unit, disp_status, card_cls, status_cls, pill_cls))
            continue

        # Raw Tesseract line — show as observation if meaningful
        if len(line) > 8 and not re.match(r'^[\d\s.:\-/]+$', line):
            # Skip obvious noise
            noise_kw = ['page ', 'www.', 'tel:', 'fax:', 'email:', 'date:', 'ref:', 'lab:']
            if not any(n in line.lower() for n in noise_kw):
                observations.append(line)

    if not metrics and not diagnoses and not observations:
        st.info("No health data found. Try a clearer image or use manual entry.")
        return

    # ── Count status summary ──────────────────────────────────────────────────
    n_abnormal   = sum(1 for m in metrics if m[4] == "rp-abnormal") + len(diagnoses)
    n_borderline = sum(1 for m in metrics if m[4] == "rp-borderline")
    n_normal     = sum(1 for m in metrics if m[4] == "rp-normal")
    n_total      = len(metrics)

    # ── Build HTML ─────────────────────────────────────────────────────────────
    html = '<div class="rp-wrap">'

    # Header bar
    html += (
        '<div class="rp-header">'
        '<div class="rp-header-left">'
        '<span class="rp-header-icon">🩺</span>'
        '<div>'
        '<div class="rp-header-title">Health Report Analysis</div>'
        f'<div class="rp-header-sub">{n_total} values extracted'
        + (f' · {len(observations)} clinical notes' if observations else '')
        + '</div>'
        '</div>'
        '</div>'
        '<div class="rp-summary-pills">'
    )
    if n_abnormal:
        html += f'<span class="rp-pill rp-pill-red">🔴 {n_abnormal} Abnormal</span>'
    if n_borderline:
        html += f'<span class="rp-pill rp-pill-yellow">⚠️ {n_borderline} Borderline</span>'
    if n_normal:
        html += f'<span class="rp-pill rp-pill-green">✅ {n_normal} Normal</span>'
    if not metrics:
        html += f'<span class="rp-pill rp-pill-blue">ℹ️ {len(observations)} Observations</span>'
    html += '</div></div>'

    # Body
    html += '<div class="rp-body">'

    # Diagnoses banner (if any detected)
    if diagnoses:
        pills = "".join(f'<span class="rp-diag-pill">{_h.escape(d)}</span>' for d in diagnoses)
        html += (
            '<div class="rp-diag-banner">'
            '<span class="rp-diag-icon">⚕️</span>'
            '<div class="rp-diag-content">'
            '<div class="rp-diag-title">Detected Conditions</div>'
            f'<div class="rp-diag-pills">{pills}</div>'
            '</div></div>'
        )

    # Group metrics by category
    if metrics:
        from collections import defaultdict
        groups = defaultdict(list)
        for metric in metrics:
            label = metric[0]
            group = _classify_line(label)
            groups[group].append(metric)

        # Render groups in priority order
        group_order = list(_METRIC_GROUPS.keys()) + ["Other Findings"]
        for group_name in group_order:
            if group_name not in groups:
                continue
            group_metrics = groups[group_name]

            # Group icon
            icons = {
                "Blood Glucose & Diabetes": "🩸",
                "Blood Pressure & Cardiac": "❤️",
                "Lipid Panel": "💛",
                "Blood Count": "🔬",
                "Iron & Anemia": "🧲",
                "Kidney Function": "🫘",
                "Liver Function": "🟤",
                "Thyroid": "🦋",
                "Vitamins & Minerals": "💊",
                "Diagnoses & Conditions": "⚕️",
                "Observations & Notes": "📋",
                "Other Findings": "📄",
            }
            icon = icons.get(group_name, "📄")

            html += (
                f'<div class="rp-group">'
                f'<div class="rp-group-title">{icon} {_h.escape(group_name)}</div>'
                f'<div class="rp-cards">'
            )

            for label, val, unit, disp_status, card_cls, status_cls, pill_cls in group_metrics:
                unit_span = f'<span class="rp-metric-unit">{_h.escape(unit)}</span>' if unit else ""
                html += (
                    f'<div class="rp-card {card_cls}">'
                    f'<div class="rp-metric-name">{_h.escape(label)}</div>'
                    f'<div class="rp-metric-value">{_h.escape(val)}{unit_span}</div>'
                    f'<span class="rp-metric-status {status_cls}">{_h.escape(disp_status)}</span>'
                    f'</div>'
                )

            html += '</div></div>'

    # Clinical observations
    if observations:
        html += (
            '<div class="rp-obs-section">'
            '<div class="rp-obs-title">📋 Clinical Observations & Notes</div>'
        )
        for idx, obs in enumerate(observations[:20], 1):
            # Clean up raw OCR artifacts
            cleaned = re.sub(r'^[•\|\+\-\*]\s*', '', obs).strip()
            cleaned = re.sub(r'\s{2,}', ' ', cleaned)
            if len(cleaned) < 4:
                continue
            html += (
                f'<div class="rp-obs-item">'
                f'<span class="rp-obs-num">{idx}</span>'
                f'<p class="rp-obs-text">{_h.escape(cleaned)}</p>'
                f'</div>'
            )
        html += '</div>'

    html += '</div></div>'  # close body + wrap
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOD CARD GRID  — 3 cards per row, professional layout
# ══════════════════════════════════════════════════════════════════════════════

# Category pill colours
_CAT_PILL = {
    "fruits":     ("rgba(244,114,182,.2)", "#f472b6"),
    "vegetables": ("rgba(52,211,153,.2)",  "#34d399"),
    "legumes":    ("rgba(251,191,36,.2)",  "#fbbf24"),
    "grains":     ("rgba(249,115,22,.2)",  "#fb923c"),
    "proteins":   ("rgba(129,140,248,.2)", "#818cf8"),
    "dairy":      ("rgba(56,189,248,.2)",  "#38bdf8"),
    "other":      ("rgba(167,139,250,.2)", "#a78bfa"),
}

def render_food_cards(recs: list, key_prefix: str = "fc"):
    """Render food cards in a 3-column professional grid with clean recipe display."""
    import html as _h

    COLS = 3  # 3 cards per row — readable, professional

    for row_start in range(0, len(recs), COLS):
        row = recs[row_start:row_start + COLS]
        n   = len(row)

        grid_html = (f'<div class="fc-grid" '
                     f'style="grid-template-columns:repeat({n},1fr);">')

        for item in row:
            cat      = item.get("category", "other")
            emoj     = CAT_EMOJI.get(cat, "🥗")
            cat_bg, cat_fg = _CAT_PILL.get(cat, ("rgba(167,139,250,.2)", "#a78bfa"))

            fname    = _h.escape(str(item["food"]))
            cal      = item.get("calories",  0)
            prot     = item.get("protein",   0)
            fat      = item.get("fat",       0)
            carb     = item.get("carbs",     0)
            fib      = item.get("fiber",     0)
            sod      = round(item.get("sodium",    0))
            iron     = item.get("iron",      0)
            potk     = round(item.get("potassium", 0))
            svg_src  = item.get("image_svg", "")

            # ── Category + Dataset badge ──────────────────────────────────
            ds_pill  = '<span class="fc-ds-pill">📊 Dataset</span>' if item.get("from_dataset") else ""
            cat_strip = (
                f'<div class="fc-cat-strip">'
                f'<span class="fc-cat-pill" style="background:{cat_bg};color:{cat_fg};">'
                f'{emoj} {cat.upper()}</span>{ds_pill}</div>'
            )

            # ── Macro rows ────────────────────────────────────────────────
            def mrow(label, val, unit="g", highlight=False):
                cls = ' class="macro-highlight"' if highlight else ""
                return (f'<div class="macro-row">'
                        f'<span>{label}</span>'
                        f'<b{cls}>{val}{unit}</b>'
                        f'</div>')

            macros = (
                mrow("🔥 Calories", f"{cal}", " kcal", highlight=True)
                + mrow("💪 Protein", f"{prot}")
                + mrow("🥑 Fat", f"{fat}")
                + mrow("🍞 Carbs", f"{carb}")
            )
            if fib:   macros += mrow("🌾 Fiber",   f"{fib}")
            if sod:   macros += mrow("🧂 Sodium",  f"{sod}", "mg")
            if iron:  macros += mrow("🩸 Iron",    f"{iron}", "mg")
            if potk:  macros += mrow("⚡ Potassium", f"{potk}", "mg")

            # ── Recipe accordion ──────────────────────────────────────────
            # Strip any residual HTML or code from recipe text
            raw_recipe = str(item.get("recipe", "")).strip()
            # Remove any Python artifacts (lambdas, function objects, etc.)
            if "<function" in raw_recipe or "lambda" in raw_recipe or raw_recipe.startswith("<"):
                raw_recipe = f"Steam or lightly cook {fname} until tender. Season with olive oil, lemon juice and herbs."
            rec_text  = _h.escape(raw_recipe[:700])

            rname     = str(item.get("recipe_name", "")).strip()
            if not rname or "<function" in rname:
                rname = f"How to Prepare {fname}"
            rname_esc = _h.escape(rname)

            ings      = item.get("ingredients", [])
            # Only show ingredients if they're real strings, not Python objects
            valid_ings = [str(g) for g in ings if g and "<function" not in str(g) and "lambda" not in str(g)]
            chips     = "".join(f'<span class="recipe-ing-chip">{_h.escape(g)}</span>'
                                for g in valid_ings[:8])
            ings_block = (
                f'<div class="recipe-ings-wrap">'
                f'<div class="recipe-ings-label">Ingredients</div>'
                f'<div class="recipe-ings-grid">{chips}</div>'
                f'</div>'
            ) if chips else ""

            recipe_html = (
                f'<details class="recipe-details">'
                f'<summary>View Recipe</summary>'
                f'<div class="recipe-content">'
                f'<div class="recipe-name-tag">{rname_esc}</div>'
                f'{ings_block}'
                f'<div class="recipe-steps-label">Preparation</div>'
                f'<p class="recipe-steps-text">{rec_text}</p>'
                f'</div></details>'
            )

            grid_html += (
                f'<div class="fc">'
                f'<div class="fc-img-wrap">'
                f'<img src="{svg_src}" class="fc-img" alt="{fname}" loading="lazy">'
                f'{cat_strip}'
                f'</div>'
                f'<div class="fc-body">'
                f'<div class="fc-title">{fname}</div>'
                f'<div class="fc-macros">{macros}</div>'
                f'{recipe_html}'
                f'</div>'
                f'</div>'
            )

        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

        # Add-to-Log buttons aligned under each card
        btn_cols = st.columns(n)
        for i, item in enumerate(row):
            with btn_cols[i]:
                if st.button(f"➕ Add to Log",
                             key=f"{key_prefix}_{row_start}_{i}",
                             use_container_width=True):
                    log_food(
                        st.session_state["user"],
                        item["food"],
                        item.get("calories", 0), item.get("protein", 0),
                        item.get("fat", 0),      item.get("carbs",   0),
                        item.get("fiber", 0),    item.get("sodium",  0),
                    )
                    st.success(f"✅ {item['food']} added to log!")
        st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DARK CHARTS
# ══════════════════════════════════════════════════════════════════════════════
DARK="#1e293b"; GRID="#334155"; FG="#e2e8f0"
AC=["#818cf8","#f472b6","#34d399","#fbbf24","#38bdf8","#a78bfa","#fb923c"]

def _base(w=9,h=3.5):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.tick_params(colors=FG,labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(color=GRID,linestyle="--",linewidth=0.5,alpha=0.5)
    return fig,ax

def _xlbls(ax, daily):
    lbls = daily["date"].astype(str).tolist()
    x    = list(range(len(lbls)))
    ax.set_xticks(x); ax.set_xticklabels(lbls,rotation=30,ha="right",fontsize=7)

def chart_trend(daily, tdee):
    fig,ax=_base(); x=list(range(len(daily))); y=daily["calories"].tolist()
    ax.plot(x,y,marker="o",color=AC[0],lw=2.5,markersize=7,label="Calories",zorder=3)
    ax.fill_between(x,y,alpha=0.12,color=AC[0])
    ax.axhline(tdee,color="#f87171",ls="--",lw=1.5,label=f"Target ({tdee})")
    _xlbls(ax,daily); ax.set_ylabel("Calories",color=FG)
    ax.set_title("Calorie Intake Trend",color=FG,fontsize=11,pad=8)
    ax.legend(facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=8)
    fig.tight_layout(); return fig

def chart_donut(prot, fat, carbs, fiber, sod):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.2))
    fig.patch.set_facecolor(DARK)
    for ax in (ax1,ax2):
        ax.set_facecolor(DARK); ax.tick_params(colors=FG,labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    # Donut — macros only (not sodium)
    vals   = [(prot,"Protein",AC[0]),(fat,"Fat",AC[1]),
              (carbs,"Carbs",AC[2]),(fiber,"Fiber",AC[3])]
    vv=[v for v,l,c in vals if v>0]
    ll=[l for v,l,c in vals if v>0]
    cc=[c for v,l,c in vals if v>0]
    if sum(vv)>0:
        wedges,_,pcts=ax1.pie(vv,labels=None,colors=cc,autopct="%1.1f%%",
            pctdistance=0.72,startangle=140,
            wedgeprops=dict(linewidth=2,edgecolor=DARK,width=0.55),
            textprops=dict(fontsize=9))
        for p in pcts: p.set_color("white"); p.set_fontweight("bold")
        total=sum(vv)
        ax1.text(0,0.06,f"{total:.0f}g",ha="center",va="center",
                 color=FG,fontsize=12,fontweight="bold")
        ax1.text(0,-0.18,"total macros",ha="center",va="center",
                 color="#64748b",fontsize=8)
        patches=[mpatches.Patch(color=c,label=f"{l}  {v:.1f}g")
                 for (v,l,c) in vals if v>0]
        ax1.legend(handles=patches,loc="lower center",bbox_to_anchor=(0.5,-0.22),
                   ncol=2,facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=7)
    ax1.set_title("Macro Split",color=FG,fontsize=11,pad=8)

    # Sodium bar
    sod_limit=2300
    sod_pct=min(sod/sod_limit,1.0)
    col="#34d399" if sod<=1500 else ("#fbbf24" if sod<=2300 else "#f87171")
    ax2.barh(["Sodium"],[ sod],color=col,height=0.4)
    ax2.barh(["Sodium"],[max(0,sod_limit-sod)],left=[min(sod,sod_limit)],
             color="#1e293b",height=0.4)
    ax2.axvline(sod_limit,color="#f87171",ls="--",lw=1.5,label="2300mg limit")
    ax2.set_xlim(0,max(sod_limit*1.1,sod*1.1))
    ax2.set_facecolor(DARK); ax2.tick_params(colors=FG,labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax2.set_title("Daily Sodium",color=FG,fontsize=11,pad=8)
    ax2.legend(facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=8)
    ax2.text(min(sod,sod_limit*1.05),0,f"{sod:.0f}mg",
             va="center",ha="left",color=FG,fontsize=9,fontweight="bold")
    fig.tight_layout(); return fig

def chart_macros_bar(daily):
    fig,ax=_base(); x=np.arange(len(daily)); w=0.2
    for i,(col,lbl) in enumerate(zip(
            ["protein","fat","carbs","fiber"],["Protein","Fat","Carbs","Fiber"])):
        if col in daily.columns:
            ax.bar(x+(i-1.5)*w,daily[col],w,label=lbl,color=AC[i],zorder=3)
    _xlbls(ax,daily); ax.set_ylabel("grams",color=FG)
    ax.set_title("Daily Macros — Protein / Fat / Carbs / Fiber",color=FG,fontsize=11,pad=8)
    ax.legend(facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=8)
    fig.tight_layout(); return fig

def chart_fiber_sodium(daily):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.2))
    fig.patch.set_facecolor(DARK)
    for ax in (ax1,ax2):
        ax.set_facecolor(DARK); ax.tick_params(colors=FG,labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID,linestyle="--",linewidth=0.5,alpha=0.5)
    x=list(range(len(daily)))
    # Fiber
    ax1.bar(x,daily["fiber"],color=AC[3],width=0.5,zorder=3)
    ax1.axhline(25,color="#fbbf24",ls="--",lw=1.4,label="Goal 25g/day")
    _xlbls(ax1,daily); ax1.set_ylabel("Fiber (g)",color=FG)
    ax1.set_title("Daily Fiber Intake",color=FG,fontsize=10,pad=6)
    ax1.legend(facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=7)
    # Sodium
    colors_s=[AC[4] if v<=2300 else "#f87171" for v in daily["sodium"]]
    ax2.bar(x,daily["sodium"],color=colors_s,width=0.5,zorder=3)
    ax2.axhline(2300,color="#f87171",ls="--",lw=1.4,label="Limit 2300mg")
    _xlbls(ax2,daily); ax2.set_ylabel("Sodium (mg)",color=FG)
    ax2.set_title("Daily Sodium Intake",color=FG,fontsize=10,pad=6)
    ax2.legend(facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=7)
    fig.tight_layout(); return fig

def chart_cal_bars(daily, tdee):
    fig,ax=_base(9,2.8); x=range(len(daily)); cs=daily["calories"].tolist()
    colors=[AC[0] if c<=tdee else "#f87171" for c in cs]
    ax.bar(x,cs,color=colors,width=0.55,zorder=3)
    ax.axhline(tdee,color="#f87171",ls="--",lw=1.4,label=f"Target ({tdee} kcal)")
    _xlbls(ax,daily); ax.set_ylabel("Calories",color=FG)
    ax.set_title("Calories vs Daily Target",color=FG,fontsize=11,pad=8)
    ax.legend(facecolor=DARK,edgecolor=GRID,labelcolor=FG,fontsize=8)
    fig.tight_layout(); return fig

# ══════════════════════════════════════════════════════════════════════════════
#  AI CHAT
# ══════════════════════════════════════════════════════════════════════════════
def _get_key(k):
    try: v=st.secrets.get(k,""); return v if v else ""
    except Exception: pass
    return os.environ.get(k,"")

def call_ai(messages, condition, bmi_val, bmi_cat, markers, gender="", age=0):
    groq_key=_get_key("GROQ_API_KEY"); gem_key=_get_key("GEMINI_API_KEY")
    system=(f"You are NutriAI, an expert clinical nutritionist and health coach.\n"
            f"User profile — BMI: {bmi_val} ({bmi_cat}), Gender: {gender}, Age: {age}, "
            f"Condition: {condition}, Markers: {json.dumps(markers) if markers else 'None'}.\n"
            f"Give personalised evidence-based nutrition advice. Use bullet points. "
            f"Keep responses 150–300 words. Always relate to their specific condition.")
    if groq_key:
        try:
            r=requests.post("https://api.groq.com/openai/v1/chat/completions",
                json={"model":"llama-3.3-70b-versatile",
                      "messages":[{"role":"system","content":system}]+messages,
                      "max_tokens":900,"temperature":0.7},
                headers={"Authorization":f"Bearer {groq_key}","Content-Type":"application/json"},
                timeout=30)
            d=r.json()
            if r.status_code==200: return d["choices"][0]["message"]["content"]
        except Exception: pass
    if gem_key:
        try:
            gmsgs=[{"role":"user" if m["role"]=="user" else "model",
                    "parts":[{"text":m["content"]}]} for m in messages]
            r=requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={gem_key}",
                json={"system_instruction":{"parts":[{"text":system}]},
                      "contents":gmsgs,"generationConfig":{"maxOutputTokens":900,"temperature":0.7}},
                headers={"Content-Type":"application/json"},timeout=30)
            d=r.json()
            if r.status_code==200: return d["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e: return f"⚠️ Connection error: {e}"
    return "⚠️ **No AI key configured.** Add `GROQ_API_KEY` to `.streamlit/secrets.toml`.\nGet a free key at **https://console.groq.com**"

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════
def auth():
    _,col,_=st.columns([1,2,1])
    with col:
        st.markdown("""<div style="text-align:center;padding:40px 0 24px;">
          <div style="font-size:3.5rem;">🥗</div>
          <h1 style="font-size:1.9rem;background:linear-gradient(135deg,#818cf8,#a78bfa);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
            NutriAI</h1>
          <p style="color:#64748b;margin-top:6px;font-size:.9rem;">
            AI-Powered Personalised Nutrition</p></div>""", unsafe_allow_html=True)
        users=load_json(USER_FILE)
        t1,t2=st.tabs(["🔐 Login","📝 Register"])
        with t1:
            u=st.text_input("Username",key="lu"); p=st.text_input("Password",type="password",key="lp")
            if st.button("Login",use_container_width=True):
                if u in users and users[u]==p:
                    st.session_state["user"]=u; st.rerun()
                else: st.error("Invalid username or password.")
        with t2:
            nu=st.text_input("New Username",key="ru"); np_=st.text_input("New Password",type="password",key="rp"); nc=st.text_input("Confirm",type="password",key="rc")
            if st.button("Create Account",use_container_width=True):
                if not nu or not np_: st.error("Fill all fields.")
                elif np_!=nc: st.error("Passwords do not match.")
                elif nu in users: st.warning("Username exists.")
                else: users[nu]=np_; save_json(USER_FILE,users); st.success("✅ Account created! Please login.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
def page_recommender():
    st.markdown("# 🥗 AI Food Recommender")
    st.caption("Dataset-driven personalised recommendations based on your health condition, BMI and gender.")

    # ── Profile inputs (persist across navigation) ────────────────────────────
    with st.expander("👤 Personal Details", expanded=not ss("recs_ready")):
        c1,c2,c3,c4=st.columns(4)
        with c1: age    = st.number_input("Age",        5,  100, ss("profile_age",30),    key="r_age")
        with c2: gender = st.selectbox("Gender",["Male","Female","Other"],
                                        index=["Male","Female","Other"].index(ss("profile_gender","Male")),
                                        key="r_gender")
        with c3: height = st.number_input("Height (cm)", 50.,250., ss("profile_height",165.), step=0.5,key="r_height")
        with c4: weight = st.number_input("Weight (kg)", 10.,300., ss("profile_weight",65.),  step=0.5,key="r_weight")
        activity = st.select_slider("Activity Level",
            ["sedentary","light","moderate","active","very active"],
            value=ss("profile_activity","moderate"),key="r_activity")

    bmi     = calculate_bmi(height, weight)
    bmi_cat = bmi_category(bmi)
    il,ih   = ideal_weight_range(height)
    tdee    = daily_calorie_needs(weight, height, age, gender, activity)

    # Persist profile
    for k,v in [("profile_age",age),("profile_gender",gender),("profile_height",height),
                ("profile_weight",weight),("profile_activity",activity),
                ("bmi",bmi),("bmi_cat",bmi_cat),("tdee",tdee)]:
        _persist(k,v)

    for col,(val,lbl) in zip(st.columns(4),[
        (bmi,"BMI Score"),(bmi_cat,"BMI Category"),(f"{il}–{ih} kg","Ideal Weight"),(f"{tdee} kcal","Daily Need")]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Health Report")
    method=st.radio("Input method",
        ["📤 Upload Medical Report Image","✏️ Enter Values Manually"],horizontal=True)

    text_lines=[]; data_provided=False

    if method=="📤 Upload Medical Report Image":
        report=st.file_uploader("Upload (JPG / PNG)",type=["jpg","jpeg","png"])
        if report:
            data_provided=True
            with st.spinner("🔍 Scanning report with AI OCR…"):
                text_lines=extract_text(report)
            _persist("ocr_lines", text_lines)
            if text_lines:
                st.success(f"✅ Report scanned — {len(text_lines)} data points found")
                with st.expander("🩺 View Extracted Report", expanded=True):
                    render_ocr_panel(text_lines)
            else:
                st.warning("No data detected. Try manual entry.")
        elif ss("ocr_lines"):
            # Restore from previous scan
            text_lines=ss("ocr_lines"); data_provided=True
            with st.expander("🩺 Previous Report (retained)", expanded=False):
                render_ocr_panel(text_lines)
    else:
        c1,c2,c3=st.columns(3)
        with c1:
            glucose    =st.number_input("Glucose (mg/dL)",     0.,1000.,0.)
            cholesterol=st.number_input("Cholesterol (mg/dL)", 0.,1000.,0.)
        with c2:
            systolic   =st.number_input("Systolic BP (mmHg)",  0.,300., 0.)
            diastolic  =st.number_input("Diastolic BP (mmHg)", 0.,200., 0.)
        with c3:
            hemoglobin =st.number_input("Hemoglobin (g/dL)",   0.,25.,  0.,step=0.1)
            hba1c      =st.number_input("HbA1c (%)",           0.,20.,  0.,step=0.1)
        if glucose     >0: text_lines.append(f"glucose: {glucose}");                data_provided=True
        if cholesterol >0: text_lines.append(f"cholesterol: {cholesterol}");        data_provided=True
        if systolic>0 and diastolic>0:
            text_lines.append(f"blood pressure: {int(systolic)}/{int(diastolic)} mmhg"); data_provided=True
        if hemoglobin  >0: text_lines.append(f"hemoglobin: {hemoglobin}");          data_provided=True
        if hba1c       >0: text_lines.append(f"hba1c: {hba1c}");                    data_provided=True
        if text_lines: _persist("ocr_lines", text_lines)
        if not data_provided:
            st.info("ℹ️ Enter at least one value above.")

    if data_provided:
        if st.button("🔬 Analyse & Get Recommendations", use_container_width=True):
            with st.spinner("🤖 Analysing and personalising recommendations from dataset…"):
                condition,all_conds,markers=detect_condition(text_lines, bmi)
                recs=recommend_food(condition, top_n=15,
                                    gender=gender, age=age, bmi_val=bmi)
            # Persist ALL results in session state
            for k,v in [("condition",condition),("all_conditions",all_conds),
                        ("markers",markers),("recommendations",recs),("recs_ready",True)]:
                _persist(k,v)
    else:
        st.button("🔬 Analyse & Get Recommendations", use_container_width=True, disabled=True)

    # Render results (persisted across navigations)
    if ss("recommendations"):
        cond  = ss("condition","Normal")
        all_c = ss("all_conditions",[cond])
        mkrs  = ss("markers",{})
        recs  = ss("recommendations",[])
        g     = ss("profile_gender","")
        a     = ss("profile_age","")

        st.markdown("---")
        ca,cb=st.columns([2,1])
        with ca:
            st.markdown("**🩺 Detected Conditions:**")
            st.markdown(" ".join(condition_badge(c) for c in all_c),unsafe_allow_html=True)
        with cb:
            real_mkrs={k:v for k,v in mkrs.items() if k!="BMI"}
            if real_mkrs:
                st.markdown("**Key Markers:**")
                for k,v in list(real_mkrs.items())[:5]:
                    st.markdown(f"• **{k}**: {v}")

        crit = CONDITION_CRITERIA
        desc = crit.get(cond,{}).get("description","")
        st.markdown(
            f"### 🍽 Personalised Recommendations — *{cond}*\n"
            f"<span style='font-size:.76rem;color:#64748b;'>"
            f"Tailored for {g} · Age {a} · BMI {bmi:.1f} · {desc}</span>",
            unsafe_allow_html=True)
        st.caption("📊 Marked foods come from your CSV dataset. 🟣 Curated medically-validated seeds come first.")

        ds_count  = sum(1 for r in recs if r.get("from_dataset"))
        cur_count = len(recs) - ds_count
        ic1,ic2,ic3=st.columns(3)
        with ic1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(recs)}</div><div class="metric-lbl">Foods Shown</div></div>',unsafe_allow_html=True)
        with ic2: st.markdown(f'<div class="metric-card"><div class="metric-val">{cur_count}</div><div class="metric-lbl">🟣 Curated Seeds</div></div>',unsafe_allow_html=True)
        with ic3: st.markdown(f'<div class="metric-card"><div class="metric-val">{ds_count}</div><div class="metric-lbl">📊 From Dataset</div></div>',unsafe_allow_html=True)

        render_food_cards(recs, key_prefix="rec")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: WELLNESS GOALS
# ══════════════════════════════════════════════════════════════════════════════
def page_wellness():
    st.markdown("# 💪 Wellness Goals")
    st.caption("Dataset-driven nutrition plans for specific fitness and health goals — personalised to your body metrics.")

    with st.expander("👤 Your Profile",expanded=not ss("goal_computed")):
        c1,c2,c3,c4=st.columns(4)
        with c1: g_age   =st.number_input("Age",5,100,ss("profile_age",28),key="g_age")
        with c2: g_gender=st.selectbox("Gender",["Male","Female","Other"],
                                        index=["Male","Female","Other"].index(ss("profile_gender","Male")),key="g_gen")
        with c3: g_ht    =st.number_input("Height (cm)",50.,250.,ss("profile_height",170.),step=0.5,key="g_ht")
        with c4: g_wt    =st.number_input("Weight (kg)",10.,300.,ss("profile_weight",70.),step=0.5,key="g_wt")
        g_act=st.select_slider("Activity",["sedentary","light","moderate","active","very active"],
                                value=ss("profile_activity","moderate"),key="g_act")

    g_bmi  =calculate_bmi(g_ht,g_wt)
    g_cat  =bmi_category(g_bmi)
    g_tdee =daily_calorie_needs(g_wt,g_ht,g_age,g_gender,g_act)
    g_il,g_ih=ideal_weight_range(g_ht)

    for k,v in [("profile_age",g_age),("profile_gender",g_gender),
                ("profile_height",g_ht),("profile_weight",g_wt),
                ("profile_activity",g_act),("bmi",g_bmi),("bmi_cat",g_cat),("tdee",g_tdee)]:
        _persist(k,v)

    for col,(val,lbl) in zip(st.columns(5),[
        (g_bmi,"BMI"),(g_cat,"Category"),(f"{g_il}–{g_ih} kg","Ideal Wt"),
        (f"{g_tdee} kcal","TDEE"),(g_gender,"Gender")]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    goal_labels=["⚖️ Weight Loss","📈 Weight Gain","💪 Gym & Muscle","✨ Skin Health"]
    goal_keys  =["Weight Loss",    "Weight Gain",    "Gym & Muscle",   "Skin Health"]
    tabs=st.tabs(goal_labels)

    for tab_idx, (tab, goal_key) in enumerate(zip(tabs, goal_keys)):
        with tab:
            # Use full sanitized key + index — prevents "Weight Loss" vs "Weight Gain" collision
            safe_key = re.sub(r'[^a-zA-Z0-9]', '_', goal_key) + f"_t{tab_idx}"
            meta=GOAL_CRITERIA[goal_key]
            color={"Weight Loss":"#34d399","Weight Gain":"#f472b6",
                   "Gym & Muscle":"#818cf8","Skin Health":"#fbbf24"}[goal_key]
            cal_target=meta["cal_formula"](g_tdee)

            st.markdown(
                f'<div class="goal-hero">'
                f'<div style="font-size:1.1rem;font-weight:800;color:{color};margin-bottom:6px;">{goal_key} Plan</div>'
                f'<div style="color:#94a3b8;font-size:.85rem;margin-bottom:14px;">{meta["description"]}</div>'
                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">'
                f'<div class="goal-stat"><div class="sv">{g_tdee}</div><div class="sl">Maintenance kcal</div></div>'
                f'<div class="goal-stat"><div class="sv" style="font-size:.95rem;color:{color};">{cal_target} kcal/day</div><div class="sl">Your Target</div></div>'
                f'<div class="goal-stat"><div class="sv" style="font-size:.8rem;color:#94a3b8;">{meta["macro_split"]}</div><div class="sl">Macro Split</div></div>'
                f'</div></div>',unsafe_allow_html=True)

            col_left,col_right=st.columns([3,1])
            with col_right:
                st.markdown("**💡 Expert Tips**")
                for tip in meta["tips"]:
                    st.markdown(f'<div class="tip-card">{tip}</div>',unsafe_allow_html=True)
            with col_left:
                st.markdown(f"**🍽 Foods for {goal_key}**")
                st.caption(f"Personalised for {g_gender} · Age {g_age} · BMI {g_bmi:.1f} — combining curated + dataset sources")
                cache_key=f"goal_recs_{goal_key}_{g_gender}_{g_age}_{g_bmi:.0f}"
                if ss(cache_key) is None:
                    with st.spinner(f"Loading {goal_key} foods from dataset…"):
                        goal_recs=recommend_food_for_goal(
                            goal_key, top_n=15,
                            gender=g_gender, age=g_age, bmi_val=g_bmi)
                    _persist(cache_key, goal_recs)
                else:
                    goal_recs=ss(cache_key)
                render_food_cards(goal_recs, key_prefix=safe_key)

            # Avg macro summary
            st.markdown("---")
            if goal_recs:
                mc1,mc2,mc3,mc4,mc5=st.columns(5)
                avg_c=round(sum(r["calories"] for r in goal_recs)/len(goal_recs),1)
                avg_p=round(sum(r["protein"]  for r in goal_recs)/len(goal_recs),1)
                avg_f=round(sum(r["fat"]       for r in goal_recs)/len(goal_recs),1)
                avg_b=round(sum(r["carbs"]     for r in goal_recs)/len(goal_recs),1)
                avg_fib=round(sum(r["fiber"]   for r in goal_recs)/len(goal_recs),1)
                for col,(val,lbl) in zip([mc1,mc2,mc3,mc4,mc5],[
                    (f"{avg_c} kcal","Avg Calories"),(f"{avg_p}g","Avg Protein"),
                    (f"{avg_f}g","Avg Fat"),(f"{avg_b}g","Avg Carbs"),(f"{avg_fib}g","Avg Fiber")]):
                    with col:
                        st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.2rem;">{val}</div>'
                                    f'<div class="metric-lbl">{lbl} per 100g</div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SKIN HEALTH (Natural Skincare Recipes)
# ══════════════════════════════════════════════════════════════════════════════
def page_skin():
    import html as _h
    st.markdown("# ✨ Skin Health Centre")
    st.caption("Evidence-based natural skincare routines and diet recommendations for every skin concern.")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1b4b,#0f172a);border:1px solid #4f46e5;
    border-radius:16px;padding:18px 24px;margin-bottom:20px;">
    <div style="font-size:.95rem;color:#a5b4fc;font-weight:700;margin-bottom:6px;">
    🌿 How this works</div>
    <div style="color:#94a3b8;font-size:.84rem;line-height:1.7;">
    Each skin concern tab provides: <b>Foods to eat</b> for internal nourishment + 
    <b>Natural topical recipes</b> made from kitchen ingredients. The recipes are 
    backed by clinical research on each active compound.
    </div></div>""", unsafe_allow_html=True)

    tab_labels = list(SKINCARE_GOALS.keys())
    tabs = st.tabs(tab_labels)

    for tab, (goal_name, goal_data) in zip(tabs, SKINCARE_GOALS.items()):
        with tab:
            color = goal_data["color"]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#1e293b,#0f172a);'
                f'border:1px solid #334155;border-radius:14px;padding:16px 20px;margin-bottom:16px;">'
                f'<div style="font-size:1rem;font-weight:800;color:{color};margin-bottom:4px;">'
                f'{goal_name}</div>'
                f'<div style="color:#94a3b8;font-size:.85rem;margin-bottom:10px;">{goal_data["desc"]}</div>'
                f'<div style="color:#64748b;font-size:.8rem;font-weight:600;">Key Nutrients: '
                f'<span style="color:{color};">{goal_data["key_nutrients"]}</span></div>'
                f'</div>', unsafe_allow_html=True)

            col_food, col_recipe = st.columns([1, 2])

            with col_food:
                st.markdown("**🥗 Foods to Eat**")
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #334155;'
                    f'border-radius:12px;padding:12px 16px;margin-bottom:10px;">'
                    f'<div style="font-size:.78rem;color:#64748b;margin-bottom:8px;">🍽 Eat these consistently:</div>'
                    + "".join(
                        f'<div style="display:flex;align-items:center;gap:8px;'
                        f'padding:5px 0;border-bottom:1px solid #1e293b;">'
                        f'<span style="font-size:1.1rem;">{CAT_EMOJI.get(_get_cat_for_food(f),"🥗")}</span>'
                        f'<span style="color:#e2e8f0;font-size:.83rem;font-weight:600;">{f}</span>'
                        f'</div>'
                        for f in goal_data["foods_to_eat"]
                    )
                    + f'</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="tip-card">'
                    f'<b>💡 Dietary Tip:</b><br>{goal_data["eat_tip"]}</div>',
                    unsafe_allow_html=True)

            with col_recipe:
                st.markdown("**🧪 Natural Skincare Recipes**")
                for recipe in goal_data["recipes"]:
                    with st.expander(f"{recipe['emoji']} {recipe['name']} — {recipe['frequency']}",
                                     expanded=False):
                        ings_html = "".join(
                            f'<span class="skin-ingredient">{_h.escape(ing)}</span>'
                            for ing in recipe["ingredients"]
                        )
                        st.markdown(
                            f'<div class="skin-recipe">'
                            f'<div style="font-size:.78rem;font-weight:700;color:#818cf8;margin-bottom:8px;">Ingredients</div>'
                            f'<div style="margin-bottom:12px;">{ings_html}</div>'
                            f'<div style="font-size:.78rem;font-weight:700;color:#34d399;margin-bottom:6px;">How to Use</div>'
                            f'<div style="font-size:.8rem;color:#94a3b8;line-height:1.6;margin-bottom:10px;">'
                            f'{_h.escape(recipe["how_to"])}</div>'
                            f'<div style="font-size:.78rem;font-weight:700;color:#fbbf24;margin-bottom:6px;">Why It Works</div>'
                            f'<div style="font-size:.8rem;color:#94a3b8;line-height:1.6;">'
                            f'{_h.escape(recipe["why_it_works"])}</div>'
                            f'<div style="margin-top:10px;display:flex;align-items:center;gap:8px;">'
                            f'<span style="background:#1e1b4b;color:#818cf8;border-radius:6px;'
                            f'padding:3px 10px;font-size:.72rem;font-weight:700;">🕐 {recipe["frequency"]}</span>'
                            f'</div></div>', unsafe_allow_html=True)

def _get_cat_for_food(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["berry","berries","kiwi","pomegranate","mango","orange","lemon","avocado","watermelon","strawberr","apricot"]): return "fruits"
    if any(k in n for k in ["spinach","broccoli","kale","tomato","carrot","sweet potato","pepper","asparagus","cucumber"]): return "vegetables"
    if any(k in n for k in ["salmon","walnut","egg","almond","seed","tuna","sardine","mackerel"]): return "proteins"
    if any(k in n for k in ["yogurt","cheese","milk"]): return "dairy"
    if any(k in n for k in ["lentil","chickpea","bean","edamame"]): return "legumes"
    if any(k in n for k in ["oat","quinoa","rice","barley"]): return "grains"
    return "other"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("# 📊 Nutrition Dashboard")
    user=st.session_state["user"]
    logs=get_user_logs(user)
    if not logs:
        st.info("📭 No food logs yet. Go to Recommender → ➕ Add to Log.")
        return

    df=pd.DataFrame(logs)
    df["date"]=pd.to_datetime(df["date"]).dt.date
    for col in ["calories","protein","fat","carbs","fiber","sodium"]:
        if col not in df.columns: df[col]=0.0
        df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0)

    period=st.radio("Period",["Today","This Week","This Month","All Time"],horizontal=True)
    today=datetime.now().date()
    if   period=="Today":      df_f=df[df["date"]==today]
    elif period=="This Week":  df_f=df[df["date"]>=today-timedelta(days=today.weekday())]
    elif period=="This Month": df_f=df[df["date"]>=today.replace(day=1)]
    else:                      df_f=df.copy()
    if df_f.empty: df_f=df.copy()

    tc   =round(df_f["calories"].sum())
    tp   =round(df_f["protein"].sum(),1)
    tf_  =round(df_f["fat"].sum(),1)
    tb   =round(df_f["carbs"].sum(),1)
    tfib =round(df_f["fiber"].sum(),1)
    tsod =round(df_f["sodium"].sum(),1)
    tdee =ss("tdee",2000)

    # Summary metrics
    for col,(val,lbl) in zip(st.columns(6),[
        (f"{tc}","🔥 Calories"),(f"{tp}g","💪 Protein"),(f"{tf_}g","🥑 Fat"),
        (f"{tb}g","🍞 Carbs"),(f"{tfib}g","🌾 Fiber"),(f"{len(df_f)}","🍽 Items")]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)

    # Progress bars
    sod_pct=min(int(tsod/2300*100),100)
    sod_col="#34d399" if tsod<=1500 else ("#fbbf24" if tsod<=2300 else "#f87171")
    cal_pct=min(tc/tdee,1.0)
    cal_col="#34d399" if cal_pct<0.85 else ("#fbbf24" if cal_pct<1.0 else "#f87171")
    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0 14px;">'
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:12px 16px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="color:#94a3b8;font-size:.83rem;">🧂 Sodium</span>'
        f'<span style="color:#e2e8f0;font-weight:700;font-size:.83rem;">{tsod}mg / 2300mg ({sod_pct}%)</span></div>'
        f'<div style="background:#0f172a;border-radius:999px;height:8px;">'
        f'<div style="background:{sod_col};width:{sod_pct}%;height:8px;border-radius:999px;"></div></div></div>'
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:12px 16px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="color:#94a3b8;font-size:.83rem;">🔥 Calories</span>'
        f'<span style="color:#e2e8f0;font-weight:700;font-size:.83rem;">{tc} / {tdee} kcal ({int(cal_pct*100)}%)</span></div>'
        f'<div style="background:#0f172a;border-radius:999px;height:8px;">'
        f'<div style="background:{cal_col};width:{int(cal_pct*100)}%;height:8px;border-radius:999px;"></div></div></div>'
        f'</div>',unsafe_allow_html=True)

    daily=df_f.groupby("date").agg(
        calories=("calories","sum"),protein=("protein","sum"),
        fat=("fat","sum"),carbs=("carbs","sum"),
        fiber=("fiber","sum"),sodium=("sodium","sum")
    ).reset_index().sort_values("date")

    tabs=st.tabs(["📈 Trend","🥧 Macros","📊 Daily Macros","🌾 Fiber & Sodium","🍽 Log"])
    with tabs[0]:
        st.markdown('<div class="chart-box">',unsafe_allow_html=True)
        fig=chart_trend(daily,tdee); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown('</div>',unsafe_allow_html=True)
    with tabs[1]:
        ca,cb=st.columns([1,1])
        with ca:
            st.markdown('<div class="chart-box">',unsafe_allow_html=True)
            fig=chart_donut(tp,tf_,tb,tfib,tsod); st.pyplot(fig,use_container_width=True); plt.close(fig)
            st.markdown('</div>',unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="chart-box">',unsafe_allow_html=True)
            total_g=tp+tf_+tb or 1
            for lbl,val,clr,sub in [
                ("💪 Protein",tp,AC[0],f"{int(tp/total_g*100)}% of macros"),
                ("🥑 Fat",tf_,AC[1],f"{int(tf_/total_g*100)}% of macros"),
                ("🍞 Carbs",tb,AC[2],f"{int(tb/total_g*100)}% of macros"),
                ("🌾 Fiber",tfib,AC[3],"Goal: 25g/day"),
                ("🧂 Sodium",tsod,AC[4],f"{int(tsod/2300*100)}% of 2300mg limit"),
            ]:
                max_v=2300 if "Sodium" in lbl else total_g
                bp=min(int(val/max_v*100),100)
                unit="mg" if "Sodium" in lbl else "g"
                st.markdown(
                    f'<div style="margin-bottom:9px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:#94a3b8;font-size:.83rem;">{lbl}</span>'
                    f'<span style="color:#e2e8f0;font-weight:700;font-size:.83rem;">'
                    f'{val}{unit} <span style="color:#475569;font-size:.74rem;">({sub})</span></span></div>'
                    f'<div style="background:#0f172a;border-radius:999px;height:7px;margin-top:3px;">'
                    f'<div style="background:{clr};width:{bp}%;height:7px;border-radius:999px;"></div>'
                    f'</div></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
    with tabs[2]:
        st.markdown('<div class="chart-box">',unsafe_allow_html=True)
        fig=chart_macros_bar(daily); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="chart-box" style="margin-top:10px;">',unsafe_allow_html=True)
        fig=chart_cal_bars(daily,tdee); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown('</div>',unsafe_allow_html=True)
    with tabs[3]:
        st.markdown('<div class="chart-box">',unsafe_allow_html=True)
        fig=chart_fiber_sodium(daily); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.markdown('</div>',unsafe_allow_html=True)
    with tabs[4]:
        cols=[c for c in ["date","food","calories","protein","fat","carbs","fiber","sodium"] if c in df_f.columns]
        rename={"date":"Date","food":"Food","calories":"Calories (kcal)","protein":"Protein (g)",
                "fat":"Fat (g)","carbs":"Carbs (g)","fiber":"Fiber (g)","sodium":"Sodium (mg)"}
        disp=df_f[cols].copy()
        disp.columns=[rename.get(c,c) for c in cols]
        st.dataframe(disp.sort_values("Date",ascending=False),use_container_width=True,hide_index=True)
        if st.button("🗑 Clear My Logs"):
            all_data=[d for d in load_json(LOG_FILE) if d.get("user")!=user]
            save_json(LOG_FILE,all_data); st.success("Logs cleared!"); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
def page_assistant():
    st.markdown("# 🧠 NutriAI Assistant")
    has_g=bool(_get_key("GROQ_API_KEY")); has_m=bool(_get_key("GEMINI_API_KEY"))
    if not has_g and not has_m:
        st.error("⚠️ No AI key configured.")
        st.markdown("Add to `.streamlit/secrets.toml`:\n```\nGROQ_API_KEY = \"gsk_...\"\n```\nGet free at **https://console.groq.com**")
        return

    provider="Groq (Llama 3.3 70B)" if has_g else "Google Gemini 1.5 Flash"
    st.success(f"✅ Connected via **{provider}**")

    condition=ss("condition","General"); bmi_val=ss("bmi",22.0)
    bmi_cat=ss("bmi_cat","Normal"); markers=ss("markers",{})
    gender=ss("profile_gender",""); age=ss("profile_age",0)
    all_c=ss("all_conditions",[condition])

    st.markdown(
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;'
        f'padding:12px 18px;margin-bottom:14px;">'
        f'<div style="color:#94a3b8;font-size:.82rem;">'
        f'BMI: <b style="color:#e2e8f0;">{bmi_val} ({bmi_cat})</b>'
        f' · Gender: <b style="color:#e2e8f0;">{gender}</b>'
        f' · Age: <b style="color:#e2e8f0;">{age}</b></div>'
        + " ".join(condition_badge(c) for c in all_c)
        + '</div>',unsafe_allow_html=True)

    st.markdown("**💬 Quick Questions:**")
    suggs=[f"Best foods for {condition}?","Create a 7-day meal plan for me.",
           f"Foods I must avoid with {condition}?","How can I improve my BMI naturally?",
           "Best breakfast, lunch and dinner?","What nutrients am I likely lacking?"]
    sc=st.columns(3)
    for i,s in enumerate(suggs):
        with sc[i%3]:
            if st.button(s,key=f"sq_{i}",use_container_width=True):
                st.session_state.setdefault("chat_messages",[])
                st.session_state["chat_messages"].append({"role":"user","content":s})
                with st.spinner("🤖 Thinking…"):
                    reply=call_ai(st.session_state["chat_messages"],condition,bmi_val,bmi_cat,markers,gender,age)
                st.session_state["chat_messages"].append({"role":"assistant","content":reply})
                st.rerun()

    st.markdown("---")
    if "chat_messages" not in st.session_state: st.session_state["chat_messages"]=[]
    if not st.session_state["chat_messages"]:
        st.markdown('<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;'
                    'padding:14px 18px;color:#94a3b8;font-size:.9rem;">'
                    '👋 Hello! I\'m <b style="color:#a5b4fc;">NutriAI</b>, your AI nutrition coach. '
                    'Ask me about meal plans, recipes, or how to manage your health condition through food.</div>',
                    unsafe_allow_html=True)
    for msg in st.session_state["chat_messages"]:
        if msg["role"]=="user":
            st.markdown(f'<div style="background:#312e81;color:#e0e7ff;border-radius:18px 18px 4px 18px;'
                        f'padding:12px 16px;margin:6px 0;max-width:78%;float:right;clear:both;font-size:.92rem;">'
                        f'🧑 {msg["content"]}</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="clear:both;color:#94a3b8;font-size:.8rem;margin-top:10px;font-weight:700;">🤖 NutriAI</div>',unsafe_allow_html=True)
            st.markdown(msg["content"])

    st.markdown("---")
    ic,bc=st.columns([5,1])
    with ic: user_input=st.text_input("Message",label_visibility="collapsed",key="chat_input",placeholder="Ask about your nutrition, meal plans, or condition…")
    with bc: send=st.button("Send 🚀",use_container_width=True)
    if send and user_input.strip():
        st.session_state["chat_messages"].append({"role":"user","content":user_input})
        with st.spinner("🤖 Thinking…"):
            reply=call_ai(st.session_state["chat_messages"],condition,bmi_val,bmi_cat,markers,gender,age)
        st.session_state["chat_messages"].append({"role":"assistant","content":reply})
        st.rerun()
    if st.session_state.get("chat_messages"):
        if st.button("🗑 Clear Chat"): st.session_state["chat_messages"]=[]; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if "user" not in st.session_state:
    auth()
else:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:14px 0 10px;">'
            '<div style="font-size:2rem;">🥗</div>'
            '<div style="font-weight:800;font-size:1.05rem;color:#a5b4fc;">NutriAI</div>'
            '<div style="font-size:.72rem;color:#475569;">AI Nutrition Recommender</div>'
            '</div>',unsafe_allow_html=True)
        st.markdown(f"👤 **{st.session_state['user']}**")
        # Show persisted condition badge (survives navigation)
        if ss("condition"):
            st.markdown(condition_badge(ss("condition")),unsafe_allow_html=True)
            bmi=ss("bmi","—"); bcat=ss("bmi_cat","—"); tdee=ss("tdee","—")
            st.markdown(f'<span style="color:#64748b;font-size:.8rem;">BMI: <b style="color:#e2e8f0;">{bmi} ({bcat})</b> · TDEE: <b style="color:#e2e8f0;">{tdee} kcal</b></span>',unsafe_allow_html=True)
        st.markdown("---")
        page=st.selectbox("Navigate",
            ["🥗 Recommender","💪 Wellness Goals","✨ Skin Health","📊 Dashboard","🧠 AI Assistant"],
            label_visibility="collapsed")
        st.markdown("---")
        has_g=bool(_get_key("GROQ_API_KEY")); has_m=bool(_get_key("GEMINI_API_KEY"))
        if has_g or has_m:
            st.markdown(f'<div style="background:#064e3b;border-radius:8px;padding:8px 12px;font-size:.75rem;color:#6ee7b7;">✅ AI: {"Groq" if has_g else "Gemini"} connected</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#450a0a;border-radius:8px;padding:8px 12px;font-size:.75rem;color:#fca5a5;">⚠️ AI: No key — see Assistant page</div>',unsafe_allow_html=True)
        st.markdown("---")
        # Quick status
        recs_count = len(ss("recommendations") or [])
        logs_count = len(get_user_logs(st.session_state["user"]))
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:10px 12px;">'
            f'<div style="color:#818cf8;font-size:.78rem;font-weight:700;margin-bottom:8px;">📊 Session Status</div>'
            f'<div style="color:#64748b;font-size:.74rem;line-height:1.8;">'
            f'🥗 Recommendations: <b style="color:#e2e8f0;">{recs_count}</b><br>'
            f'📝 Food logs today: <b style="color:#e2e8f0;">{logs_count}</b><br>'
            f'🔬 Condition: <b style="color:#e2e8f0;">{ss("condition","Not set")}</b>'
            f'</div></div>',unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚪 Logout",use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        st.markdown('<div style="font-size:.67rem;color:#334155;text-align:center;margin-top:10px;">v5.0 · NutriAI</div>',unsafe_allow_html=True)

    if   "Recommender"   in page: page_recommender()
    elif "Wellness"      in page: page_wellness()
    elif "Skin Health"   in page: page_skin()
    elif "Dashboard"     in page: page_dashboard()
    elif "Assistant"     in page: page_assistant()