"""
ocr_reader.py  — Medical report extraction using Claude Vision (Anthropic API)

Replaces the old Tesseract-based approach. Claude Vision reads any real-world
health report (blood test, lab report, discharge summary, etc.) and extracts
ALL clinically-relevant data points in a structured, clean format.

Falls back to Tesseract if the Anthropic API key is unavailable.
"""

import re
import os
import base64
import platform
from io import BytesIO


# ─────────────────────────────────────────────────────────────────────────────
#  CLAUDE VISION (primary — works on any health report, any language)
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key():
    """Read Anthropic API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _image_to_base64(image_file):
    """Convert uploaded file / PIL Image / path to (base64_str, media_type)."""
    try:
        from PIL import Image as _PIL
        if isinstance(image_file, _PIL.Image.Image):
            buf = BytesIO()
            image_file.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode(), "image/png"
    except ImportError:
        pass

    if hasattr(image_file, "read"):
        raw  = image_file.read()
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        name = getattr(image_file, "name", "") or ""
        ext  = name.rsplit(".", 1)[-1].lower() if "." in name else "png"
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                "pdf":"application/pdf","gif":"image/gif","webp":"image/webp"}
        return base64.b64encode(raw).decode(), mime.get(ext, "image/png")

    if isinstance(image_file, (str, bytes)):
        path = str(image_file)
        ext  = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode(), mime.get(ext, "image/png")

    raise ValueError(f"Unsupported image_file type: {type(image_file)}")


_CLAUDE_PROMPT = """You are a medical data extraction specialist. Carefully read this health or medical report image and extract ALL clinically relevant information — both numeric lab values AND text observations, clinical notes, and diagnoses.

Return a clean list of findings, one per line. Use these formats:

For lab values with numbers:
  Label: Value Unit — ✅ Normal / ⚠️ Borderline / 🔴 Abnormal

For diagnoses mentioned in the report:
  Diagnosis: [Condition Name] — 🔴 Detected

For clinical observations and notes:
  Observation: [Brief text of the finding]

For tests mentioned but without a result value:
  Test Mentioned: [Test Name]

Rules:
- Extract EVERYTHING clinically relevant: glucose, HbA1c, blood pressure, hemoglobin, cholesterol (total/LDL/HDL), triglycerides, ferritin, serum iron, creatinine, urea, uric acid, TSH, T3/T4, WBC, RBC, platelets, SGOT, SGPT, bilirubin, vitamin D, vitamin B12, calcium, sodium, potassium, eGFR
- Also extract: any mentioned disease names, diagnostic impressions, clinical notes, recommendations, abnormal flags (H/L/CRITICAL), family history mentions, risk factors
- For numeric values use status: ✅ Normal, ⚠️ Borderline, 🔴 Abnormal/High/Low
- If the report has a HIGH/LOW/CRITICAL marker next to a value, always use 🔴
- Do NOT include: patient name, age, date of birth, address, doctor name, hospital name, report ID number, collection date/time
- Do NOT invent values not visible in the image
- If this is not any kind of medical report: respond with exactly NOT_A_MEDICAL_REPORT
- Return ONLY the list. No preamble, no explanation, no markdown formatting."""


def _extract_with_claude(image_file):
    """Use Claude Vision to extract all health data. Returns list or None."""
    import requests as _req

    api_key = _get_api_key()
    if not api_key:
        return None  # fall back to Tesseract

    try:
        b64, media_type = _image_to_base64(image_file)
    except Exception as e:
        return [f"Image conversion error: {e}"]

    # PDFs need conversion to image first
    if media_type == "application/pdf":
        try:
            import fitz
            if hasattr(image_file, "read"):
                image_file.seek(0)
                raw = image_file.read()
            else:
                with open(str(image_file), "rb") as f:
                    raw = f.read()
            doc  = fitz.open(stream=raw, filetype="pdf")
            page = doc[0]
            pix  = page.get_pixmap(dpi=200)
            buf  = BytesIO(pix.tobytes("png"))
            b64  = base64.b64encode(buf.getvalue()).decode()
            media_type = "image/png"
        except ImportError:
            return ["PDF support requires PyMuPDF (pip install pymupdf). Use JPG/PNG instead."]
        except Exception as e:
            return [f"PDF read error: {e}"]

    payload = {
        "model": "claude-opus-4-6",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": media_type,
                                              "data": b64}},
                {"type": "text", "text": _CLAUDE_PROMPT},
            ],
        }],
    }

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={"x-api-key": api_key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            timeout=45,
        )
        data = resp.json()
        if resp.status_code != 200:
            err = data.get("error", {}).get("message", str(data))
            return [f"Claude Vision API error ({resp.status_code}): {err}"]

        raw_text = data["content"][0]["text"].strip()

        if raw_text == "NOT_A_MEDICAL_REPORT":
            return ["⚠️ This does not appear to be a medical report. Please upload a health or lab report image."]

        lines    = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        findings = [ln for ln in lines if ":" in ln]
        return findings if findings else ["No health data detected — try manual entry below."]

    except Exception as e:
        return [f"Claude Vision error: {e} — try manual entry below."]


# ─────────────────────────────────────────────────────────────────────────────
#  TESSERACT FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pytesseract
    from PIL import Image as _PILImg, ImageEnhance, ImageOps
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


_TESS_PATTERNS = [
    ("Glucose",          r'(?:fasting\s+blood\s+)?glucose[\w\s,]*?(\d{2,3}\.?\d*)', "mg/dL", 126, 70),
    ("Glucose",          r'\bfbs[\s:]+(\d{2,3}\.?\d*)',                              "mg/dL", 126, 70),
    ("HbA1c",            r'hba?1c[^0-9]*(\d+\.?\d*)',                                "%",     6.5, None),
    ("Haemoglobin",      r'(?:haemo?globin|hemoglobin)\s*[:\(]?\s*(\d+\.?\d*)',      "g/dL",  None, 12),
    ("Haemoglobin",      r'\bhb\b[\s:]+(\d+\.?\d*)',                                 "g/dL",  None, 12),
    ("Total Cholesterol",r'(?:total\s+)?cholesterol[\s:]+(\d{2,3}\.?\d*)',           "mg/dL", 200, None),
    ("LDL Cholesterol",  r'ldl[\s\w]*?cholesterol[\s:]+(\d{2,3}\.?\d*)',             "mg/dL", 100, None),
    ("HDL Cholesterol",  r'hdl[\s\w]*?cholesterol[\s:]+(\d{1,3}\.?\d*)',             "mg/dL", None, 40),
    ("Triglycerides",    r'triglycerides?[\s:]+(\d{2,3}\.?\d*)',                     "mg/dL", 150, None),
    ("Ferritin",         r'ferritin[\s:]+(\d+\.?\d*)',                               "ng/mL", None, 12),
    ("Serum Iron",       r'serum\s+iron[\s:]+(\d+\.?\d*)',                           "µg/dL", None, 60),
    ("Creatinine",       r'creatinine[\s:]+(\d+\.?\d*)',                             "mg/dL", 1.2, None),
    ("TSH",              r'\btsh[\s:]+(\d+\.?\d*)',                                  "mIU/L", 4.5, 0.4),
    ("Vitamin D",        r'(?:vitamin\s+d|25-oh)[\s:]+(\d+\.?\d*)',                  "ng/mL", None, 20),
    ("Vitamin B12",      r'(?:vitamin\s+b12|b-12|b12)[\s:]+(\d+\.?\d*)',             "pg/mL", None, 200),
]

_STATUS_MAP = {"HIGH":"⚠️ HIGH","LOW":"⚠️ LOW","OK":"✅ Normal"}


def _best_ocr(img):
    if img.mode == "RGBA": img = img.convert("RGB")
    elif img.mode not in ("RGB","L"): img = img.convert("RGB")
    results = []
    for attempt in [
        lambda i: pytesseract.image_to_string(i, config="--psm 6 --oem 3"),
        lambda i: pytesseract.image_to_string(ImageEnhance.Contrast(i.convert("L")).enhance(2.5), config="--psm 6 --oem 3"),
        lambda i: pytesseract.image_to_string(i.resize((i.size[0]*2,i.size[1]*2),_PILImg.LANCZOS).convert("L"), config="--psm 6 --oem 3"),
        lambda i: pytesseract.image_to_string(ImageEnhance.Contrast(ImageOps.invert(i.convert("L"))).enhance(2.0), config="--psm 6 --oem 3"),
    ]:
        try:
            t = attempt(img)
            if t.strip(): results.append(t)
        except Exception:
            pass
    return max(results, key=len) if results else ""


def _tesseract_extract(image_file):
    """
    Extract text from the report using Tesseract OCR.
    Returns raw meaningful lines from the image (like the old behavior)
    so the user can see all extracted content — plus any structured findings.
    """
    if not TESSERACT_AVAILABLE:
        return ["Tesseract OCR not installed. Add ANTHROPIC_API_KEY to secrets.toml for Claude Vision OCR, or install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"]
    try:
        img      = _PILImg.open(image_file)
        raw_text = _best_ocr(img)
        if not raw_text.strip():
            return ["No text detected in image — try a clearer photo or manual entry."]

        # ── Step 1: Return ALL meaningful raw lines (old behaviour) ─────────
        raw_lines = _filter_raw_lines(raw_text)

        # ── Step 2: Append structured findings if any numeric values found ──
        structured = _parse_tess(raw_text)
        # Only add structured lines that aren't already visible in raw lines
        extra = [s for s in structured if not any(
            s.split(":")[0].lower() in ln.lower() for ln in raw_lines
        )]

        result = raw_lines + (["──────────────────"] + extra if extra else [])
        return result if result else ["No health data detected — try manual entry below."]

    except Exception as e:
        return [f"OCR Error: {e} — try manual entry below."]


# ── Noise words that indicate a line is not useful ───────────────────────────
_NOISE_WORDS = {
    "page", "of", "ref", "reference", "laboratory", "lab", "report",
    "tel", "phone", "fax", "email", "www", "http", "address", "signature",
    "authorised", "authorized", "printed", "generated", "remarks",
}

def _filter_raw_lines(raw_text: str) -> list:
    """
    Split OCR raw text into non-empty, meaningful lines.
    Filters out:
    - Lines shorter than 4 characters
    - Lines that are only digits/punctuation
    - Lines that are clearly header/footer noise
    Keeps up to 20 lines.
    """
    lines   = raw_text.splitlines()
    results = []
    seen    = set()

    for line in lines:
        line = line.strip()
        # Skip empty or too-short
        if len(line) < 4:
            continue
        # Skip lines that are only numbers, dashes, underscores, dots
        if re.match(r'^[\d\s\.\-\_\|\/\\\:]+$', line):
            continue
        # Skip obvious noise (contains too many noise words relative to total words)
        words = [w.strip(':.,-;()') for w in line.lower().split()]
        if words and sum(1 for w in words if w in _NOISE_WORDS) / len(words) > 0.5:
            continue
        # Skip page number lines like "Page 1 of 3"
        if re.match(r'^page\s+\d+\s+of\s+\d+$', line.lower()):
            continue
        # Skip phone/fax/email lines
        if re.match(r'^(tel|fax|ph|phone|email|mob|mobile|www)[\s:.]', line.lower()):
            continue
        # Deduplicate
        key = re.sub(r'\s+', '', line.lower())
        if key in seen:
            continue
        seen.add(key)
        results.append(line)
        if len(results) >= 20:
            break

    return results


def _parse_tess(raw_text):
    if not raw_text.strip(): return []
    text     = re.sub(r'\s+', ' ', raw_text.lower()).strip()
    findings = []
    seen     = set()

    bp = re.search(r'(\b1[2-9]\d|[2-9]\d\d)\s*/\s*([5-9]\d|1\d{2})', text)
    if not bp: bp = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})\s*mmhg', text)
    if bp and "Blood Pressure" not in seen:
        s, d = int(bp.group(1)), int(bp.group(2))
        if 70 <= s <= 250 and 40 <= d <= 150:
            seen.add("Blood Pressure")
            status = "🔴 High" if (s>=140 or d>=90) else ("⚠️ Borderline" if s>=120 else "✅ Normal")
            findings.append(f"Blood Pressure: {s}/{d} mmHg — {status}")

    for label, pattern, unit, high_t, low_t in _TESS_PATTERNS:
        if label in seen: continue
        m = re.search(pattern, text)
        if not m: continue
        try: val = float(m.group(1))
        except (IndexError, ValueError): continue
        status = (_STATUS_MAP["HIGH"] if high_t and val >= high_t
                  else _STATUS_MAP["LOW"] if low_t and val < low_t
                  else _STATUS_MAP["OK"])
        seen.add(label)
        findings.append(f"{label}: {val} {unit} — {status}")
        if len(findings) >= 15: break

    for label, kws, flag in [
        ("Diabetes",    ["diabetes","diabetic"],                "🔴 Detected"),
        ("Hypertension",["hypertension"],                       "🔴 Detected"),
        ("Anaemia",     ["anaemia","anemia","iron deficiency"], "🔴 Detected"),
        ("Thyroid",     ["hypothyroid","hyperthyroid","thyroid"],"🔴 Detected"),
    ]:
        if label not in seen and any(k in text for k in kws):
            seen.add(label)
            findings.append(f"Diagnosis: {label} — {flag}")

    return findings


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(image_file):
    """
    Extract all clinically relevant findings from a medical report image.

    Uses Claude Vision (Anthropic API) if ANTHROPIC_API_KEY is set —
    works on any type of health report with high accuracy.
    Falls back to Tesseract OCR if no API key is configured.

    Returns a list of formatted finding strings, e.g.:
      ["Glucose: 148.0 mg/dL — 🔴 Abnormal",
       "Blood Pressure: 158/98 mmHg — 🔴 High",
       "Diagnosis: Hypertension — 🔴 Detected"]
    """
    if hasattr(image_file, "seek"):
        image_file.seek(0)

    api_key = _get_api_key()
    if api_key:
        result = _extract_with_claude(image_file)
        if result is not None:
            return result

    if hasattr(image_file, "seek"):
        image_file.seek(0)
    return _tesseract_extract(image_file)
