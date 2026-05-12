"""
condition_detector.py — Detect health conditions from OCR-extracted medical data.

Works with both:
  - Claude Vision output  (structured "Label: Value Unit — Status" lines)
  - Tesseract plain-text output (raw OCR dump)
  - Manual entry text lines (e.g. "glucose: 148", "blood pressure: 158/98 mmhg")

Returns: (primary_condition, all_conditions_list, markers_dict)
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
#  STRUCTURED LINE PARSER  (for Claude Vision output)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_structured_lines(lines):
    """
    Parse structured findings list produced by Claude Vision, e.g.:
      "Glucose: 148.0 mg/dL — 🔴 Abnormal"
      "Blood Pressure: 140/90 mmHg — 🔴 High"
      "Diagnosis: Hypertension — 🔴 Detected"

    Returns dict:  { "Glucose": 148.0, "Blood Pressure": (140,90), "Diagnosis": ["Hypertension"], ... }
    """
    parsed = {}
    diagnoses = []

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        # Split on first colon
        label_part, rest = line.split(":", 1)
        label = label_part.strip().lower()
        # Remove status suffix after "—"
        value_part = rest.split("—")[0].strip()

        # ── Diagnosis keyword line ─────────────────────────────────────
        if "diagnosis" in label or "detected" in rest.lower():
            diag = value_part.strip().strip("🔴").strip()
            if diag:
                diagnoses.append(diag)
            continue

        # ── Blood Pressure ────────────────────────────────────────────
        bp_m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', value_part)
        if bp_m and ("blood" in label or "bp" in label or "pressure" in label):
            parsed["blood_pressure"] = (int(bp_m.group(1)), int(bp_m.group(2)))
            continue

        # ── Numeric values ─────────────────────────────────────────────
        num_m = re.search(r'(\d+\.?\d*)', value_part)
        if not num_m:
            continue
        val = float(num_m.group(1))

        # Map label keywords → canonical keys
        if any(k in label for k in ["glucose","blood sugar","fbs","fasting blood"]):
            parsed["glucose"] = val
        elif "hba1c" in label or "hba" in label or "glycated" in label:
            parsed["hba1c"] = val
        elif "hemoglobin" in label or "haemoglobin" in label or (label.strip() == "hb"):
            parsed["hemoglobin"] = val
        elif "ldl" in label:
            parsed["ldl"] = val
        elif "hdl" in label:
            parsed["hdl"] = val
        elif "total cholesterol" in label or (label.strip() == "cholesterol"):
            parsed["cholesterol"] = val
        elif "triglyceride" in label:
            parsed["triglycerides"] = val
        elif "ferritin" in label:
            parsed["ferritin"] = val
        elif "serum iron" in label or label.strip() == "iron":
            parsed["serum_iron"] = val
        elif "creatinine" in label:
            parsed["creatinine"] = val
        elif "urea" in label or "bun" in label:
            parsed["urea"] = val
        elif "uric acid" in label:
            parsed["uric_acid"] = val
        elif "tsh" in label:
            parsed["tsh"] = val
        elif "t3" in label and "t4" not in label:
            parsed["t3"] = val
        elif "t4" in label:
            parsed["t4"] = val
        elif "vitamin d" in label or "25-oh" in label:
            parsed["vitamin_d"] = val
        elif "vitamin b12" in label or "b12" in label:
            parsed["vitamin_b12"] = val
        elif "wbc" in label or "white blood cell" in label or "leukocyte" in label:
            parsed["wbc"] = val
        elif "rbc" in label or "red blood cell" in label or "erythrocyte" in label:
            parsed["rbc"] = val
        elif "platelet" in label:
            parsed["platelets"] = val
        elif "sgpt" in label or "alt" in label:
            parsed["sgpt"] = val
        elif "sgot" in label or "ast" in label:
            parsed["sgot"] = val
        elif "bilirubin" in label:
            parsed["bilirubin"] = val
        elif "calcium" in label:
            parsed["calcium"] = val
        elif "sodium" in label:
            parsed["sodium_lab"] = val
        elif "potassium" in label:
            parsed["potassium"] = val
        elif "egfr" in label or "gfr" in label:
            parsed["egfr"] = val
        elif "albumin" in label:
            parsed["albumin"] = val

    parsed["diagnoses"] = diagnoses
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
#  RAW TEXT PARSER  (for Tesseract / manual entry fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_raw_text(text_lines):
    """Parse unstructured raw OCR text. Returns same dict format."""
    full = re.sub(r'\s+', ' ', " ".join(str(l) for l in text_lines).lower()).strip()
    parsed = {"diagnoses": []}

    def _find(patterns):
        for pat in patterns:
            m = re.search(pat, full)
            if m:
                try: return float(m.group(1))
                except Exception: pass
        return None

    glucose = _find([
        r'(?:fasting\s+blood\s+)?glucose[\w\s,]*?(\d{2,3}\.?\d*)',
        r'\bfbs[\s:]+(\d{2,3}\.?\d*)',
        r'blood\s+sugar[\s:]+(\d{2,3}\.?\d*)',
    ])
    if glucose: parsed["glucose"] = glucose

    hba1c = _find([r'hba?1c[^0-9]*(\d+\.?\d*)'])
    if hba1c: parsed["hba1c"] = hba1c

    bp_m = re.search(r'(\b1[2-9]\d|[2-9]\d\d)\s*/\s*([5-9]\d|1\d{2})', full)
    if not bp_m: bp_m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})\s*mmhg', full)
    if bp_m:
        s, d = int(bp_m.group(1)), int(bp_m.group(2))
        if 70 <= s <= 250 and 40 <= d <= 150:
            parsed["blood_pressure"] = (s, d)

    hb = _find([
        r'(?:haemoglobin|hemoglobin|h(?:a?e)?moglobin)\s*[:\(]?\s*(\d+\.?\d*)',
        r'\bhb\b[\s:]+(\d+\.?\d*)',
    ])
    if hb: parsed["hemoglobin"] = hb

    chol = _find([r'(?:total\s+)?cholesterol[\s:]+(\d{2,3}\.?\d*)'])
    if chol: parsed["cholesterol"] = chol

    ldl = _find([r'ldl[\s\w]*?cholesterol[\s:]+(\d{2,3}\.?\d*)', r'\bldl[\s:]+(\d{2,3}\.?\d*)'])
    if ldl: parsed["ldl"] = ldl

    trig = _find([r'triglycerides?[\s:]+(\d{2,3}\.?\d*)'])
    if trig: parsed["triglycerides"] = trig

    ferritin = _find([r'ferritin[\s:]+(\d+\.?\d*)'])
    if ferritin: parsed["ferritin"] = ferritin

    iron = _find([r'serum\s+iron[\s:]+(\d+\.?\d*)'])
    if iron: parsed["serum_iron"] = iron

    tsh = _find([r'\btsh[\s:]+(\d+\.?\d*)'])
    if tsh: parsed["tsh"] = tsh

    vd = _find([r'(?:vitamin\s+d|25-oh)[\s:]+(\d+\.?\d*)'])
    if vd: parsed["vitamin_d"] = vd

    b12 = _find([r'(?:vitamin\s+b12|b-12|b12)[\s:]+(\d+\.?\d*)'])
    if b12: parsed["vitamin_b12"] = b12

    # Keyword diagnoses
    kw_map = {
        "Diabetes":     ["diabetes","diabetic","hyperglycaemia","hyperglycemia"],
        "Hypertension": ["hypertension","high blood pressure"],
        "Anemia":       ["anaemia","anemia","iron deficiency","microcytic"],
        "Hypothyroid":  ["hypothyroid"],
        "Hyperthyroid": ["hyperthyroid"],
    }
    for cond, kws in kw_map.items():
        if any(k in full for k in kws):
            parsed["diagnoses"].append(cond)

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
#  CONDITION LOGIC  — applies clinical thresholds
# ─────────────────────────────────────────────────────────────────────────────

def _apply_thresholds(p, bmi):
    """
    Given parsed values dict + BMI, determine list of conditions + markers.
    Returns (conditions_list, markers_dict).
    """
    conditions = []
    markers    = {}

    # ── Glucose / Diabetes ──────────────────────────────────────────────
    glucose = p.get("glucose")
    if glucose is not None:
        markers["Glucose"] = f"{glucose} mg/dL"
        if glucose >= 126:
            conditions.append("Diabetes")
        elif glucose >= 100:
            conditions.append("Pre-Diabetes")

    hba1c = p.get("hba1c")
    if hba1c is not None:
        markers["HbA1c"] = f"{hba1c} %"
        if hba1c >= 6.5 and "Diabetes" not in conditions:
            conditions.append("Diabetes")
        elif hba1c >= 5.7 and "Pre-Diabetes" not in conditions and "Diabetes" not in conditions:
            conditions.append("Pre-Diabetes")

    # ── Blood Pressure / Hypertension ───────────────────────────────────
    bp = p.get("blood_pressure")
    if bp:
        sys_v, dia_v = bp
        markers["Blood Pressure"] = f"{sys_v}/{dia_v} mmHg"
        if sys_v >= 140 or dia_v >= 90:
            conditions.append("Hypertension")
        elif sys_v >= 120 or dia_v >= 80:
            if "Hypertension" not in conditions:
                conditions.append("Pre-Hypertension")

    # ── Cholesterol / Heart Risk ─────────────────────────────────────────
    chol = p.get("cholesterol")
    if chol is not None:
        markers["Total Cholesterol"] = f"{chol} mg/dL"
        if chol >= 240 and "Heart Risk" not in conditions:
            conditions.append("Heart Risk")
        elif chol >= 200 and "Borderline Cholesterol" not in conditions:
            conditions.append("Borderline Cholesterol")

    ldl = p.get("ldl")
    if ldl is not None:
        markers["LDL Cholesterol"] = f"{ldl} mg/dL"
        if ldl >= 130 and "Heart Risk" not in conditions:
            conditions.append("Heart Risk")

    hdl = p.get("hdl")
    if hdl is not None:
        markers["HDL Cholesterol"] = f"{hdl} mg/dL"
        if hdl < 40 and "Heart Risk" not in conditions:
            conditions.append("Heart Risk")

    trig = p.get("triglycerides")
    if trig is not None:
        markers["Triglycerides"] = f"{trig} mg/dL"
        if trig >= 200 and "Heart Risk" not in conditions:
            conditions.append("Heart Risk")
        elif trig >= 150 and "Borderline Cholesterol" not in conditions and "Heart Risk" not in conditions:
            conditions.append("Borderline Cholesterol")

    # ── Anemia ─────────────────────────────────────────────────────────
    hb = p.get("hemoglobin")
    if hb is not None:
        markers["Hemoglobin"] = f"{hb} g/dL"
        if hb < 12.0 and "Anemia" not in conditions:
            conditions.append("Anemia")

    ferritin = p.get("ferritin")
    if ferritin is not None:
        markers["Ferritin"] = f"{ferritin} ng/mL"
        if ferritin < 12 and "Anemia" not in conditions:
            conditions.append("Anemia")

    iron = p.get("serum_iron")
    if iron is not None:
        markers["Serum Iron"] = f"{iron} µg/dL"
        if iron < 60 and "Anemia" not in conditions:
            conditions.append("Anemia")

    # ── Kidney / Renal ──────────────────────────────────────────────────
    creat = p.get("creatinine")
    if creat is not None:
        markers["Creatinine"] = f"{creat} mg/dL"
        if creat > 1.2 and "Kidney Disease" not in conditions:
            conditions.append("Kidney Disease")

    egfr = p.get("egfr")
    if egfr is not None:
        markers["eGFR"] = f"{egfr} mL/min/1.73m²"
        if egfr < 60 and "Kidney Disease" not in conditions:
            conditions.append("Kidney Disease")

    # ── Liver ───────────────────────────────────────────────────────────
    sgpt = p.get("sgpt")
    sgot = p.get("sgot")
    if sgpt is not None:
        markers["SGPT/ALT"] = f"{sgpt} U/L"
        if sgpt > 56 and "Liver Disorder" not in conditions:
            conditions.append("Liver Disorder")
    if sgot is not None:
        markers["SGOT/AST"] = f"{sgot} U/L"
        if sgot > 40 and "Liver Disorder" not in conditions:
            conditions.append("Liver Disorder")

    # ── Thyroid ─────────────────────────────────────────────────────────
    tsh = p.get("tsh")
    if tsh is not None:
        markers["TSH"] = f"{tsh} mIU/L"
        if tsh > 4.5 and "Hypothyroid" not in conditions:
            conditions.append("Hypothyroid")
        elif tsh < 0.4 and "Hyperthyroid" not in conditions:
            conditions.append("Hyperthyroid")

    # ── Vitamin deficiencies ─────────────────────────────────────────────
    vd = p.get("vitamin_d")
    if vd is not None:
        markers["Vitamin D"] = f"{vd} ng/mL"
        if vd < 20 and "Vitamin D Deficiency" not in conditions:
            conditions.append("Vitamin D Deficiency")

    b12 = p.get("vitamin_b12")
    if b12 is not None:
        markers["Vitamin B12"] = f"{b12} pg/mL"
        if b12 < 200 and "Vitamin B12 Deficiency" not in conditions:
            conditions.append("Vitamin B12 Deficiency")

    # ── Keyword diagnoses from OCR ────────────────────────────────────────
    DIAG_MAP = {
        "diabetes":             "Diabetes",
        "pre-diabetes":         "Pre-Diabetes",
        "pre diabetes":         "Pre-Diabetes",
        "hypertension":         "Hypertension",
        "pre-hypertension":     "Pre-Hypertension",
        "pre hypertension":     "Pre-Hypertension",
        "anemia":               "Anemia",
        "anaemia":              "Anemia",
        "heart risk":           "Heart Risk",
        "hypothyroid":          "Hypothyroid",
        "hyperthyroid":         "Hyperthyroid",
        "kidney disease":       "Kidney Disease",
        "renal failure":        "Kidney Disease",
        "liver disorder":       "Liver Disorder",
        "obesity":              "Obesity",
        "vitamin d deficiency": "Vitamin D Deficiency",
        "vitamin b12 deficiency":"Vitamin B12 Deficiency",
    }
    for raw_diag in p.get("diagnoses", []):
        raw_lower = raw_diag.lower().strip()
        for kw, cond in DIAG_MAP.items():
            if kw in raw_lower and cond not in conditions:
                conditions.append(cond)

    # ── BMI-based conditions ─────────────────────────────────────────────
    markers["BMI"] = str(bmi)
    if bmi >= 30 and "Obesity" not in conditions:
        conditions.append("Obesity")
    elif bmi < 18.5 and "Underweight" not in conditions:
        conditions.append("Underweight")

    return conditions, markers


# ─────────────────────────────────────────────────────────────────────────────
#  PRIORITY → PRIMARY CONDITION
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY = [
    "Diabetes", "Pre-Diabetes",
    "Hypertension", "Pre-Hypertension",
    "Heart Risk", "Borderline Cholesterol",
    "Anemia",
    "Hypothyroid", "Hyperthyroid",
    "Kidney Disease", "Liver Disorder",
    "Obesity", "Underweight",
    "Vitamin D Deficiency", "Vitamin B12 Deficiency",
]

# Map detected conditions → recommender condition keys
CONDITION_ALIAS = {
    "Hypothyroid":            "Normal",   # handled via Normal + diet advice
    "Hyperthyroid":           "Normal",
    "Kidney Disease":         "Normal",
    "Liver Disorder":         "Normal",
    "Vitamin D Deficiency":   "Normal",
    "Vitamin B12 Deficiency": "Anemia",   # similar dietary needs
}


def detect_condition(text_lines, bmi):
    """
    Detect health conditions from OCR output lines and BMI.

    Handles both:
      - Structured Claude Vision output ("Label: Value Unit — Status")
      - Raw Tesseract / manual entry text

    Returns:
      primary_condition (str)  — key matching CONDITION_RULES in recommender
      all_conditions (list)    — every detected condition
      markers (dict)           — {label: "value unit"} for display
    """
    # Determine if this looks like structured Claude Vision output
    structured_lines = [
        l for l in text_lines
        if isinstance(l, str) and ":" in l and (
            "—" in l or "✅" in l or "⚠️" in l or "🔴" in l or "Detected" in l
        )
    ]
    is_structured = len(structured_lines) >= 1

    if is_structured:
        parsed = _parse_structured_lines(text_lines)
    else:
        parsed = _parse_raw_text(text_lines)

    conditions, markers = _apply_thresholds(parsed, bmi)

    # Pick primary condition
    primary = "Normal"
    for p in PRIORITY:
        if p in conditions:
            primary = p
            break

    # Map to recommender-compatible key
    primary = CONDITION_ALIAS.get(primary, primary)

    if not conditions:
        conditions = ["Normal"]

    return primary, conditions, markers
